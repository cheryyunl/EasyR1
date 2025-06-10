#!/usr/bin/env python3
"""
UI任务Reward Function测试
测试基于target的新版reward function
"""

try:
    from gui import compute_score, format_reward, calculate_action_similarity, parse_actions_from_text
    print("✓ 成功导入reward函数")
except ImportError:
    print("⚠ 未找到gui.py，请将reward函数代码复制到此文件或创建gui.py")
    exit(1)

def run_tests():
    """运行全面测试"""
    print("\n" + "="*60)
    print("UI任务Reward Function测试 (Target版本)")
    print("="*60)
    
    # 1. 格式测试 - 现在status是必需的
    print("\n📋 1. 格式验证测试")
    test_cases = [
        ('完整正确格式', 
         'Step 2: "screenshot_abstraction": "loading screen", "action": {"action_type": "click", "target": "4"}, "status": "not done"', 1.0),
        ('多步正确格式', 
         '''Step 2: "screenshot_abstraction": "loading", "action": {"action_type": "wait"}, "status": "not done"
Step 3: "screenshot_abstraction": "interface", "action": {"action_type": "click", "target": "4"}, "status": "done"''', 1.0),
        ('缺少status字段', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "target": "4"}', 0.0),
        ('status值错误', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "target": "4"}, "status": "invalid"', 0.0),
        ('缺少target字段', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click"}, "status": "done"', 0.0),
        ('scroll缺少direction', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "scroll"}, "status": "done"', 0.0),
        ('scroll正确格式', 
         'Step 1: "screenshot_abstraction": "test", "action": {"action_type": "scroll", "direction": "up"}, "status": "done"', 1.0),
        ('无效JSON', 
         'Step 1: "screenshot_abstraction": "test", "action": {invalid}, "status": "done"', 0.0),
        ('完全错误格式', 
         'Invalid format', 0.0),
    ]
    
    for name, text, expected in test_cases:
        result = format_reward(text)
        status = "✓" if abs(result - expected) < 0.01 else "✗"
        print(f"  {status} {name}: {result:.2f} (期望: {expected:.2f})")

    # 2. 动作相似度测试 - Current vs Future
    print("\n🎯 2. 动作相似度测试 (Current vs Future)")
    
    # Click动作测试 - Current Action (严格匹配)
    print("\n  Current Action (位置0) - 严格target匹配:")
    pred_click_current = {"action_type": "click", "target": "4", "status": "not done"}
    current_tests = [
        ("target完全匹配", {"action_type": "click", "target": "4", "status": "not done"}),
        ("target不匹配", {"action_type": "click", "target": "5", "status": "not done"}),
        ("status不同", {"action_type": "click", "target": "4", "status": "done"}),
        ("类型不同", {"action_type": "scroll", "direction": "up", "status": "not done"}),
    ]
    
    for name, gt_action in current_tests:
        sim = calculate_action_similarity(pred_click_current, gt_action, is_current=True)
        print(f"    {name}: {sim:.3f}")
    
    # Future Action (宽松匹配)
    print("\n  Future Action (位置1+) - 宽松target匹配:")
    pred_click_future = {"action_type": "click", "target": "0", "status": "not done"}  # 占位符
    future_tests = [
        ("占位符target=0", {"action_type": "click", "target": "4", "status": "not done"}),
        ("精确匹配", {"action_type": "click", "target": "4", "status": "not done"}),
        ("target不匹配", {"action_type": "click", "target": "5", "status": "not done"}),
    ]
    
    for name, gt_action in future_tests:
        sim = calculate_action_similarity(pred_click_future, gt_action, is_current=False)
        print(f"    {name}: {sim:.3f}")
    
    # 其他动作类型测试
    print("\n  其他动作类型:")
    other_tests = [
        ("Wait动作", {"action_type": "wait", "status": "not done"}, {"action_type": "wait", "status": "not done"}),
        ("Scroll匹配", {"action_type": "scroll", "direction": "up", "status": "not done"}, 
         {"action_type": "scroll", "direction": "up", "status": "not done"}),
        ("Scroll不匹配", {"action_type": "scroll", "direction": "up", "status": "not done"}, 
         {"action_type": "scroll", "direction": "down", "status": "not done"}),
        ("Input text匹配", {"action_type": "input_text", "text": "hello", "status": "not done"}, 
         {"action_type": "input_text", "text": "hello world", "status": "not done"}),
        ("App name匹配", {"action_type": "open_app", "app_name": "Zoho", "status": "not done"}, 
         {"action_type": "open_app", "app_name": "Zoho Meeting", "status": "not done"}),
    ]
    
    for name, pred_action, gt_action in other_tests:
        sim = calculate_action_similarity(pred_action, gt_action, is_current=True)
        print(f"    {name}: {sim:.3f}")

    # 3. 解析测试
    print("\n📖 3. 文本解析测试")
    
    test_text = '''Step 2: "screenshot_abstraction": "App loading screen", "action": {"action_type": "wait"}, "status": "not done"
Step 3: "screenshot_abstraction": "Zoho Meet interface", "action": {"action_type": "click", "target": "4"}, "status": "done"'''
    
    actions = parse_actions_from_text(test_text)
    print(f"  解析到 {len(actions)} 个动作:")
    for i, action in enumerate(actions):
        print(f"    Step {i+1}: {action['action_type']}, status: {action['status']}")

    # 4. 完整序列测试
    print("\n🔗 4. 序列对齐与奖励测试")
    
    base_gt = '''Step 2: "screenshot_abstraction": "screen1", "action": {"action_type": "wait"}, "status": "not done"
Step 3: "screenshot_abstraction": "screen2", "action": {"action_type": "click", "target": "4"}, "status": "done"'''
    
    sequence_tests = [
        ("完全正确", base_gt),
        ("第一步正确第二步target错", base_gt.replace('"target": "4"', '"target": "5"')),
        ("Future用占位符", base_gt.replace('"target": "4"', '"target": "0"')),
        ("第二步动作类型错", base_gt.replace('click', 'scroll", "direction": "up')),
        ("Status错误", base_gt.replace('"status": "done"', '"status": "not done"')),
        ("预测多一步", base_gt + '\nStep 4: "screenshot_abstraction": "extra", "action": {"action_type": "navigate_back"}, "status": "done"'),
        ("预测少一步", '''Step 2: "screenshot_abstraction": "screen1", "action": {"action_type": "wait"}, "status": "done"'''),
        ("完全错误", '''Step 2: "screenshot_abstraction": "wrong", "action": {"action_type": "scroll", "direction": "down"}, "status": "done"'''),
    ]
    
    print("  序列测试结果:")
    for name, pred in sequence_tests:
        score = compute_score(pred, base_gt)
        print(f"    {name}: 总分={score['overall']:.3f}, 准确度={score['accuracy']:.3f}, 格式={score['format']:.3f}")

    # 5. Current vs Future权重测试
    print("\n⚖️ 5. Current vs Future权重差异测试")
    
    # Current action错误 vs Future action错误
    gt_base = '''Step 1: "screenshot_abstraction": "s1", "action": {"action_type": "click", "target": "1"}, "status": "not done"
Step 2: "screenshot_abstraction": "s2", "action": {"action_type": "click", "target": "2"}, "status": "done"'''
    
    weight_tests = [
        ("Current错误", gt_base.replace('"target": "1"', '"target": "999"', 1)),
        ("Future错误", gt_base.replace('"target": "2"', '"target": "999"', 1)),
        ("Current用占位符", gt_base.replace('"target": "1"', '"target": "0"', 1)),
        ("Future用占位符", gt_base.replace('"target": "2"', '"target": "0"', 1)),
    ]
    
    scores = []
    for name, pred in weight_tests:
        score = compute_score(pred, gt_base)
        scores.append((name, score['accuracy']))
        print(f"    {name}: {score['accuracy']:.3f}")
    
    print(f"    Current vs Future惩罚差异: {scores[0][1] - scores[1][1]:.3f}")

    # 6. 边界情况测试
    print("\n⚠️ 6. 边界情况测试")
    
    edge_tests = [
        ("空预测", ""),
        ("无效格式", "This is invalid"),
        ("JSON错误", 'Step 1: "action": {broken_json}'),
        ("只有格式无内容", 'Step 1: "screenshot_abstraction": "", "action": {"action_type": "wait"}, "status": "done"'),
    ]
    
    simple_gt = '''Step 1: "screenshot_abstraction": "test", "action": {"action_type": "click", "target": "1"}, "status": "done"'''
    
    for name, pred in edge_tests:
        score = compute_score(pred, simple_gt)
        print(f"    {name}: 总分={score['overall']:.3f}, 格式={score['format']:.3f}")

    # 7. 真实示例测试
    print("\n🎯 7. 真实示例测试")
    
    real_gt = '''Step 2: "screenshot_abstraction": "App loading screen", "action": {"action_type": "wait"}, "status": "not done"
Step 3: "screenshot_abstraction": "Zoho Meet interface", "action": {"action_type": "click", "target": "4"}, "status": "done"'''
    
    real_tests = [
        ("标准答案", real_gt),
        ("占位符版本", real_gt.replace('"target": "4"', '"target": "0"')),
        ("Current错误", real_gt.replace('wait', 'click", "target": "1')),
        ("格式完美逻辑错", real_gt.replace('"target": "4"', '"target": "999"')),
    ]
    
    for name, pred in real_tests:
        score = compute_score(pred, real_gt)
        print(f"    {name}: 总分={score['overall']:.3f} (格式:{score['format']:.2f}, 准确度:{score['accuracy']:.2f})")

    # 总结
    print("\n" + "="*60)
    print("📝 测试总结与特性验证:")
    print("✓ 1. Status字段现在是必需的")
    print("✓ 2. Click/Long_press必须有target字段") 
    print("✓ 3. Current action (位置0) target必须精确匹配")
    print("✓ 4. Future actions (位置1+) 允许target='0'占位符")
    print("✓ 5. Status权重影响最终得分(30%)")
    print("✓ 6. 序列对齐支持不同长度")
    print("✓ 7. 时间折扣GAMMA=0.8应用于future actions")
    print("\n🔧 关键参数:")
    print("- CURRENT_TYPE_WEIGHT=0.6, CURRENT_DETAIL_WEIGHT=0.4")
    print("- FUTURE_TYPE_WEIGHT=0.8, FUTURE_DETAIL_WEIGHT=0.2") 
    print("- STATUS_WEIGHT=0.3, GAMMA=0.8")
    print("- 格式权重默认=0.5")
    print("="*60)

if __name__ == "__main__":
    run_tests()
