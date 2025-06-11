#!/usr/bin/env python3
"""
测试Qwen模型在Android Control数据集上的输出格式
验证是否符合reward function的期望
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import re

def test_qwen_android_output():
    print("🔍 测试Qwen在Android Control数据集上的输出")
    print("=" * 60)
    
    # 1. 加载数据集
    print("\n📊 加载数据集...")
    try:
        dataset = load_dataset("cheryyunl/android_control", split="validation")
        print(f"✅ 数据集加载成功，共{len(dataset)}个样本")
        
        # 查看第一个样本
        first_sample = dataset[0]
        print(f"\n📋 第一个样本预览:")
        print(f"Problem字段: {first_sample['problem'][:200]}...")
        print(f"Images字段: {type(first_sample.get('images', 'N/A'))}")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return
    
    # 2. 加载模型
    print(f"\n🤖 加载Qwen模型...")
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ 模型加载成功")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return
    
    # 3. 构建prompt
    system_prompt = """You MUST first think step-by-step in <think> </think> tags, then provide your final answer in <answer> </answer> tags.
Task: Given a task goal and previous steps, predict the remaining steps to complete the UI task.
**IMPORTANT RULES:**
- Screenshot contains numbered marks indicating interactive elements 
- Current step (first step you predict): Use specific mark ID from screenshot for click/long_press actions
- Future steps: Use target "0" for click/long_press actions (placeholder)
- Mark the final action step as "status": "done" and all intermediate steps have "status": "not done"
**EFFICIENCY RULES:**
- Do NOT repeat actions or create unnecessary loops
- Keep steps minimal and direct
Answer Output Format: 
Step X: "screenshot_abstraction": "brief description", "action": {...}, "status": "not done|done"
Step X+1: "screenshot_abstraction": "brief description", "action": {...}, "status": "not done|done"
Available Actions:
- click: {"action_type": "click", "target": "mark_id"}
- long_press: {"action_type": "long_press", "target": "mark_id"}
- scroll: {"action_type": "scroll", "direction": "up|down|left|right"}
- open_app: {"action_type": "open_app", "app_name": "app_name"}
- input_text: {"action_type": "input_text", "text": "text_to_type"}
- navigate_home: {"action_type": "navigate_home"}
- navigate_back: {"action_type": "navigate_back"}
- wait: {"action_type": "wait"}"""

    user_prompt = first_sample['problem']
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"\n📝 用户prompt预览:")
    print(f"{user_prompt[:300]}...")
    
    # 4. 生成回答
    print(f"\n🚀 开始推理...")
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    print(f"✅ 推理完成")
    
    # 5. 分析输出格式
    print(f"\n📄 模型完整输出:")
    print("=" * 40)
    print(response)
    print("=" * 40)
    
    # 6. 验证格式
    print(f"\n🔍 格式验证:")
    
    # 检查XML格式
    xml_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    xml_match = xml_pattern.search(response)
    print(f"✅ XML格式检查: {'✓ PASS' if xml_match else '❌ FAIL'}")
    
    if xml_match:
        # 提取answer部分
        answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
        if answer_match:
            answer_content = answer_match.group(1).strip()
            print(f"\n📋 Answer内容:")
            print(answer_content)
            
            # 检查步骤格式
            steps = re.findall(r'Step\s+\d+:', answer_content)
            print(f"\n🔢 检测到步骤数: {len(steps)}")
            print(f"步骤列表: {steps}")
            
            # 检查JSON格式
            action_matches = re.findall(r'"action":\s*(\{[^}]+\})', answer_content)
            print(f"\n🎬 检测到动作数: {len(action_matches)}")
            
            valid_actions = 0
            for i, action_str in enumerate(action_matches):
                try:
                    import json
                    action_dict = json.loads(action_str)
                    print(f"  Action {i+1}: ✓ {action_dict.get('action_type', 'unknown')}")
                    valid_actions += 1
                except:
                    print(f"  Action {i+1}: ❌ JSON格式错误")
            
            print(f"\n📊 格式总结:")
            print(f"  XML格式: {'✓' if xml_match else '❌'}")
            print(f"  步骤数量: {len(steps)}")
            print(f"  有效动作: {valid_actions}/{len(action_matches)}")
            print(f"  格式正确率: {(valid_actions/len(action_matches)*100) if action_matches else 0:.1f}%")
    
    # 7. 测试reward function兼容性
    print(f"\n🏆 Reward Function兼容性测试:")
    
    # 模拟format_reward函数
    if xml_match:
        print("✅ 通过XML格式检查")
        if answer_match and steps:
            print("✅ 通过步骤检查")
            print("✅ 应该能获得较高的format_reward分数")
        else:
            print("❌ 步骤格式可能有问题")
    else:
        print("❌ XML格式不匹配，format_reward = 0.0")
    
    print(f"\n🎯 结论:")
    if xml_match and steps and valid_actions > 0:
        print("✅ 模型输出格式基本符合预期！")
        print("✅ 应该能正常进入训练阶段")
        print("💡 如果还在循环rollout，可能是其他配置问题")
    else:
        print("❌ 模型输出格式不符合预期")
        print("💡 这可能是rollout循环的原因")

if __name__ == "__main__":
    test_qwen_android_output()
