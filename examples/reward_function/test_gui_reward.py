#!/usr/bin/env python3
try:
    from gui import compute_score, format_reward, calculate_action_similarity, parse_actions_from_text
    print("✓ 成功导入reward函数")
except ImportError:
    print("⚠ 未找到gui_reward.py，请将reward函数代码复制到此文件或创建gui_reward.py")
    exit(1)

def run_tests():
    """运行全面测试"""
    print("\n" + "="*60)
    print("GUI Agent Reward Function 测试")
    print("="*60)
    
    # 1. 格式测试
    print("\n📋 1. 格式验证测试")
    test_cases = [
        ('正确格式+status', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "x": 100}, "status": "done"', 1.0),
        ('正确格式无status', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "x": 100}', 1.0),
        ('无效JSON', 
         'Step 1: "screenshot_abstraction": "test", "action": {invalid}, "status": "done"', 0.0),
        ('错误格式', 
         'Invalid format', 0.0),
    ]
    
    for name, text, expected in test_cases:
        result = format_reward(text)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} {name}: {result:.2f} (期望: {expected:.2f})")

    # 2. 动作相似度测试
    print("\n🎯 2. 动作相似度测试")
    
    # Click动作测试
    pred_click = {"action_type": "click", "x": 100, "y": 200, "status": "not done"}
    test_actions = [
        ("完全匹配", {"action_type": "click", "x": 100, "y": 200, "status": "not done"}),
        ("坐标接近", {"action_type": "click", "x": 105, "y": 205, "status": "not done"}),
        ("status不同", {"action_type": "click", "x": 100, "y": 200, "status": "done"}),
        ("类型不同", {"action_type": "scroll", "direction": "up", "status": "not done"}),
    ]
    
    for name, gt_action in test_actions:
        sim = calculate_action_similarity(pred_click, gt_action)
        print(f"  Click {name}: {sim:.3f}")
    
    # 文本动作测试
    pred_text = {"action_type": "input_text", "text": "hello world", "status": "not done"}
    text_tests = [
        ("完全匹配", {"action_type": "input_text", "text": "hello world", "status": "not done"}),
        ("部分匹配", {"action_type": "input_text", "text": "hello", "status": "not done"}),
        ("不匹配", {"action_type": "input_text", "text": "goodbye", "status": "not done"}),
    ]
    
    for name, gt_action in text_tests:
        sim = calculate_action_similarity(pred_text, gt_action)
        print(f"  Text {name}: {sim:.3f}")

    # 3. 完整序列测试
    print("\n🔗 3. 序列对齐与奖励测试")
    
    base_gt = '''Step 1: "screenshot_abstraction": "screen1", "action": {"action_type": "click", "x": 100, "y": 200}, "status": "not done"
Step 2: "screenshot_abstraction": "screen2", "action": {"action_type": "wait"}, "status": "done"'''
    
    sequence_tests = [
        ("完全正确", base_gt),
        ("第二步错误", base_gt.replace('wait', 'scroll", "direction": "up')),
        ("预测多一步", base_gt + '\nStep 3: "screenshot_abstraction": "extra", "action": {"action_type": "navigate_back"}, "status": "done"'),
        ("预测少一步", '''Step 1: "screenshot_abstraction": "screen1", "action": {"action_type": "click", "x": 100, "y": 200}, "status": "done"'''),
        ("完全错误", '''Step 1: "screenshot_abstraction": "wrong", "action": {"action_type": "scroll", "direction": "down"}, "status": "done"'''),
    ]
    
    for name, pred in sequence_tests:
        score = compute_score(pred, base_gt)
        print(f"  {name}: 总分={score['overall']:.3f}, 准确度={score['accuracy']:.3f}, 格式={score['format']:.3f}")

    # 4. Status影响测试
    print("\n📊 4. Status权重影响测试")
    
    base_action = '"action": {"action_type": "click", "x": 100, "y": 200}'
    gt_done = f'Step 1: "screenshot_abstraction": "test", {base_action}, "status": "done"'
    
    status_tests = [
        ("Status匹配", f'Step 1: "screenshot_abstraction": "test", {base_action}, "status": "done"'),
        ("Status不匹配", f'Step 1: "screenshot_abstraction": "test", {base_action}, "status": "not done"'),
        ("Status缺失", f'Step 1: "screenshot_abstraction": "test", {base_action}'),
    ]
    
    scores = []
    for name, pred in status_tests:
        score = compute_score(pred, gt_done)
        scores.append(score['accuracy'])
        print(f"  {name}: {score['accuracy']:.3f}")
    
    print(f"  Status影响差异: {max(scores) - min(scores):.3f}")

    # 5. 边界情况测试
    print("\n⚠️  5. 边界情况测试")
    
    edge_tests = [
        ("空预测", ""),
        ("无效格式", "This is invalid"),
        ("JSON错误", 'Step 1: "action": {broken_json}'),
    ]
    
    simple_gt = '''Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "x": 100}, "status": "done"'''
    
    for name, pred in edge_tests:
        score = compute_score(pred, simple_gt)
        print(f"  {name}: {score['overall']:.3f}")

    # 总结
    print("\n" + "="*60)
    print("📝 测试总结与建议:")
    print("1. ✓ 格式验证应该正确识别有效/无效格式")
    print("2. ✓ 完全匹配动作的相似度应该接近1.0") 
    print("3. ✓ Status权重(30%)应该合理影响得分")
    print("4. ✓ 预测多/少步骤应该有适当惩罚")
    print("5. ✓ 边界情况应该优雅降级到0分")
    print("\n🔧 如果发现异常结果，请检查:")
    print("- 权重设置是否合理 (STATUS_WEIGHT=0.3)")
    print("- 坐标容忍度是否适当 (COORDINATE_TOLERANCE=0.04)")
    print("- 折扣因子是否合理 (GAMMA=0.8)")
    print("="*60)

if __name__ == "__main__":
    run_tests()
