import re
import json
import argparse
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def calculate_f1_score(predicted_str, ground_truth_str):
    """计算F1分数 - 从原代码复制"""
    predicted_str = predicted_str.replace("[", "").replace("]", "")
    ground_truth_str = ground_truth_str.replace("[", "").replace("]", "")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens) == 1 and len(ground_truth_tokens) == 1:
        predicted_token = list(predicted_tokens)[0]
        ground_truth_token = list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def parse_step_action(step_line):
    """解析step行，提取action信息"""
    try:
        # 查找action部分
        action_match = re.search(r'"action":\s*(\{[^}]+\})', step_line)
        if action_match:
            action_str = action_match.group(1)
            # 修复JSON格式（处理可能的单引号等问题）
            action_str = action_str.replace("'", '"')
            action = json.loads(action_str)
            return action
    except Exception as e:
        logger.warning(f"Failed to parse action from line: {step_line[:100]}...")
        logger.warning(f"Error: {e}")
    return None

def extract_first_step(content, section_type):
    """从output或ground_truth部分提取第一个step"""
    # 查找section标记
    section_start = content.find(f"[{section_type}]")
    if section_start == -1:
        logger.warning(f"Could not find [{section_type}] section")
        return None
    
    # 从section开始位置往后查找
    remaining_content = content[section_start:]
    
    # 查找下一个section的开始（作为结束边界）
    next_section = float('inf')
    for next_tag in ['[prompt]', '[output]', '[ground_truth]', '[score]']:
        if next_tag != f"[{section_type}]":
            pos = remaining_content.find(next_tag, 1)  # 从位置1开始查找，避免找到当前section
            if pos != -1:
                next_section = min(next_section, pos)
    
    if next_section == float('inf'):
        section_content = remaining_content
    else:
        section_content = remaining_content[:next_section]
    
    logger.debug(f"Section [{section_type}] content preview: {section_content[:200]}...")
    
    # 在section内容中查找所有Step，然后选择编号最小的
    lines = section_content.split('\n')
    steps_found = []
    
    for line in lines:
        line = line.strip()
        if line.startswith("Step") and "action" in line:
            # 提取step编号
            import re
            step_match = re.match(r'Step\s+(\d+):', line)
            if step_match:
                step_num = int(step_match.group(1))
                action = parse_step_action(line)
                if action:
                    steps_found.append((step_num, action, line))
    
    if not steps_found:
        logger.warning(f"No valid steps found in [{section_type}] section")
        return None
    
    # 选择编号最小的step
    steps_found.sort(key=lambda x: x[0])  # 按step编号排序
    first_step = steps_found[0]
    logger.debug(f"Found {len(steps_found)} steps, using Step {first_step[0]}: {first_step[2][:100]}...")
    
    return first_step[1]  # 返回action

def evaluate_target_match(pred_action, gt_action):
    """评估target是否匹配"""
    action_type = pred_action.get('action_type')
    
    if action_type in ['click', 'long_press']:
        # Grounding任务：比较target数字ID
        pred_target = pred_action.get('target', '')
        gt_target = gt_action.get('target', '')
        return pred_target == gt_target
    
    elif action_type in ['input_text']:
        # 文本任务：使用F1分数
        pred_text = pred_action.get('text', '')
        gt_text = gt_action.get('text', '')
        f1_score = calculate_f1_score(pred_text, gt_text)
        return f1_score >= 0.5
    
    elif action_type in ['scroll']:
        # 滚动任务：比较方向
        pred_direction = pred_action.get('direction', '')
        gt_direction = gt_action.get('direction', '')
        return pred_direction == gt_direction
    
    elif action_type in ['open_app']:
        # 打开应用：比较应用名
        pred_app = pred_action.get('app_name', '')
        gt_app = gt_action.get('app_name', '')
        f1_score = calculate_f1_score(pred_app, gt_app)
        return f1_score >= 0.5
    
    else:
        # 其他动作类型，如果没有额外参数则认为匹配
        return True

def evaluate_log_file(log_file_path):
    """评估整个log文件"""
    with open(log_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 按[prompt]分割测试案例
    test_cases = []
    sections = content.split('[prompt]')
    
    for section in sections[1:]:  # 跳过第一个空的部分
        # 每个section应该包含完整的测试案例
        if '[output]' in section and '[ground_truth]' in section:
            test_cases.append('[prompt]' + section)  # 重新加上[prompt]标记
    
    total_cases = 0
    type_success = 0
    step_success = 0
    
    results = []
    
    for i, case_content in enumerate(test_cases):
        # 提取第一个step
        pred_action = extract_first_step(case_content, 'output')
        gt_action = extract_first_step(case_content, 'ground_truth')
        
        if pred_action is None or gt_action is None:
            logger.warning(f"Case {i+1}: Could not extract actions")
            logger.warning(f"  Pred action: {pred_action}")
            logger.warning(f"  GT action: {gt_action}")
            continue
        
        total_cases += 1
        
        # 检查action type匹配
        pred_type = pred_action.get('action_type')
        gt_type = gt_action.get('action_type')
        type_match = (pred_type == gt_type)
        
        step_match = False
        if type_match:
            type_success += 1
            
            # 检查target匹配
            target_match = evaluate_target_match(pred_action, gt_action)
            if target_match:
                step_success += 1
                step_match = True
        
        # 记录结果
        result = {
            'case_id': i + 1,
            'pred_action': pred_action,
            'gt_action': gt_action,
            'type_match': type_match,
            'target_match': target_match if type_match else False,
            'step_match': step_match
        }
        results.append(result)
        
        logger.info(f"Case {i+1}: Type={type_match}, Step={step_match}")
        logger.info(f"  Pred: {pred_action}")
        logger.info(f"  GT:   {gt_action}")
        
        # 如果target不匹配，输出详细信息
        if type_match and not target_match:
            logger.info(f"  Target mismatch details:")
            if pred_type in ['click', 'long_press']:
                logger.info(f"    Pred target: '{pred_action.get('target', '')}', GT target: '{gt_action.get('target', '')}'")
            elif pred_type == 'input_text':
                pred_text = pred_action.get('text', '')
                gt_text = gt_action.get('text', '')
                f1 = calculate_f1_score(pred_text, gt_text)
                logger.info(f"    Pred text: '{pred_text}', GT text: '{gt_text}', F1: {f1:.3f}")
        
        logger.info("")
    
    # 计算最终指标
    type_success_rate = type_success / total_cases if total_cases > 0 else 0
    step_success_rate = step_success / total_cases if total_cases > 0 else 0
    
    logger.info("=" * 50)
    logger.info(f"FINAL RESULTS:")
    logger.info(f"Total test cases: {total_cases}")
    logger.info(f"Type success: {type_success}/{total_cases} = {type_success_rate:.4f}")
    logger.info(f"Step success: {step_success}/{total_cases} = {step_success_rate:.4f}")
    logger.info("=" * 50)
    
    return {
        'total_cases': total_cases,
        'type_success': type_success,
        'step_success': step_success,
        'type_success_rate': type_success_rate,
        'step_success_rate': step_success_rate,
        'detailed_results': results
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate first step matching in log files')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--output_file', type=str, help='Path to save detailed results (JSON)')
    
    args = parser.parse_args()
    
    # 设置logging
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # 评估
    results = evaluate_log_file(args.log_file)
    
    # 保存详细结果
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Detailed results saved to: {args.output_file}")

if __name__ == '__main__':
    main()
