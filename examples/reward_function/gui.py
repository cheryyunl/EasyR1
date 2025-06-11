# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import json
import numpy as np
from typing import Dict, List, Tuple

# Configuration
GAMMA = 0.8                          # Discount factor for future actions
COORDINATE_TOLERANCE = 0.04          # 4% tolerance for coordinate-based actions
FUTURE_TOLERANCE_MULTIPLIER = 1.5    # Looser tolerance for future actions

# Reward weights: current vs future actions
CURRENT_TYPE_WEIGHT = 0.6    # Current action type importance
CURRENT_DETAIL_WEIGHT = 0.4  # Current action detail importance  
FUTURE_TYPE_WEIGHT = 0.8     # Future action type importance (higher!)
FUTURE_DETAIL_WEIGHT = 0.2   # Future action detail importance (lower!)

def format_reward(predict: str) -> float:
    """
    Check if prediction follows correct format:
    <think>...</think><answer>Step X: ...</answer>
    """
    xml_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(xml_pattern, predict):
        return 0.0
    
    answer_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    if not answer_match:
        return 0.0
    
    answer_content = answer_match.group(1).strip()
    all_steps = re.findall(r'Step\s+\d+:', answer_content) 
    if len(all_steps) == 0:
        return 0.0
    
    valid_steps = 0
    for step_match in re.finditer(r'(Step\s+\d+:.*?)(?=Step\s+\d+:|$)', answer_content, re.DOTALL):
        step_text = step_match.group(1).strip()
        
        try:
            has_screenshot = re.search(r'"screenshot_abstraction":\s*"[^"]*"', step_text)
            has_action = re.search(r'"action":\s*(\{[^}]+\})', step_text)
            has_status = re.search(r'"status":\s*"(done|not done)"', step_text)
            
            if not (has_screenshot and has_action and has_status):
                continue
            
            action_match = has_action
            action_dict = json.loads(action_match.group(1))
            
            if 'action_type' not in action_dict:
                continue
            
            action_type = action_dict['action_type']
            
            if action_type in ['click', 'long_press']:
                if 'target' not in action_dict:
                    continue
            elif action_type == 'scroll':
                if 'direction' not in action_dict or action_dict['direction'] not in ['up', 'down', 'left', 'right']:
                    continue
            elif action_type == 'open_app':
                if 'app_name' not in action_dict:
                    continue
            elif action_type == 'input_text':
                if 'text' not in action_dict:
                    continue
            elif action_type in ['navigate_home', 'navigate_back', 'wait']:
                pass
            else:
                continue
            
            valid_steps += 1
            
        except Exception:
            continue
    
    return valid_steps / len(all_steps)

def parse_actions_from_text(text: str) -> List[dict]:
    """Extract action sequence from text, including status."""
    # Pre-process the text to handle format issues
    text = re.sub(r"\s*(<|>|/)\s*", r"\1", text)
    
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        content = answer_match.group(1).strip()
    else:
        content = text
    
    actions = []
    
    # Parse each step individually 
    for step_match in re.finditer(r'(Step\s+\d+:.*?)(?=Step\s+\d+:|$)', content, re.DOTALL):
        step_text = step_match.group(1).strip()
        
        try:
            # Extract action and status from this step
            action_match = re.search(r'"action":\s*(\{[^}]+\})', step_text)
            status_match = re.search(r'"status":\s*"([^"]+)"', step_text)
            
            if action_match and status_match:
                action_dict = json.loads(action_match.group(1))
                action_dict['status'] = status_match.group(1)
                actions.append(action_dict)
        except:
            continue
    return actions

def calculate_action_similarity(pred_action: dict, gt_action: dict, is_current: bool = True) -> float:
    """Calculate similarity score between two actions (0-1)."""
    # Type must match
    if pred_action.get('action_type') != gt_action.get('action_type'):
        return 0.0
    
    action_type = pred_action.get('action_type')
    
    # Calculate detail similarity
    detail_similarity = 0.0
    
    if action_type in ['click', 'long_press']:
        if is_current:
            # For current action: strict target matching
            pred_target = str(pred_action.get('target', ''))
            gt_target = str(gt_action.get('target', ''))
            detail_similarity = 1.0 if pred_target == gt_target else 0.0
        else:
            # For future actions: lenient target matching (allow placeholder "0")
            pred_target = str(pred_action.get('target', ''))
            gt_target = str(gt_action.get('target', ''))
            
            # Accept if prediction uses placeholder "0" or matches exactly
            if pred_target == "0" or pred_target == gt_target:
                detail_similarity = 1.0
            else:
                detail_similarity = 0.0
        
    elif action_type == 'scroll':
        detail_similarity = 1.0 if pred_action.get('direction') == gt_action.get('direction') else 0.0
        
    elif action_type in ['input_text', 'open_app']:
        # Text similarity (substring matching)
        pred_text = str(pred_action.get('text', '') or pred_action.get('app_name', '')).lower().strip()
        gt_text = str(gt_action.get('text', '') or gt_action.get('app_name', '')).lower().strip()
        
        detail_similarity = 1.0 if (pred_text in gt_text) or (gt_text in pred_text) else 0.0
        
    elif action_type in ['navigate_home', 'navigate_back', 'wait']:
        detail_similarity = 1.0
    else:
        # Unknown action type
        detail_similarity = 0.0

    # Status similarity
    pred_status = pred_action.get('status', 'not done')
    gt_status = gt_action.get('status', 'not done')
    status_similarity = 1.0 if pred_status == gt_status else 0.0

    STATUS_WEIGHT = 0.3  
    combined_similarity = (1 - STATUS_WEIGHT) * detail_similarity + STATUS_WEIGHT * status_similarity
    
    return combined_similarity

def find_best_alignment(pred_actions: List[dict], gt_actions: List[dict]) -> List[Tuple[int, int, float]]:
    """
    Find best action alignment using greedy matching.
    Returns: [(pred_idx, gt_idx, similarity_score), ...]
    """
    # If same length, use position-based matching (encourage exact order)
    if len(pred_actions) == len(gt_actions):
        alignments = []
        for i in range(len(pred_actions)):
            is_current = (i == 0)
            similarity = calculate_action_similarity(pred_actions[i], gt_actions[i], is_current)
            alignments.append((i, i, similarity))
        return alignments
    
    # Different lengths: greedy matching with position preference
    alignments = []
    used_gt_indices = set()
    
    for pred_idx, pred_action in enumerate(pred_actions):
        best_gt_idx = None
        best_score = 0.0
        is_current = (pred_idx == 0)
        
        for gt_idx, gt_action in enumerate(gt_actions):
            if gt_idx in used_gt_indices:
                continue
                
            similarity = calculate_action_similarity(pred_action, gt_action, is_current)
            
            # Position penalty: prefer maintaining order
            position_penalty = abs(pred_idx - gt_idx) * 0.1
            adjusted_score = max(0, similarity - position_penalty)
            
            if adjusted_score > best_score and adjusted_score > 0.5:
                best_gt_idx = gt_idx
                best_score = adjusted_score
        
        if best_gt_idx is not None:
            alignments.append((pred_idx, best_gt_idx, best_score))
            used_gt_indices.add(best_gt_idx)
    
    return alignments

def count_repetitions(actions: List[dict]) -> int:
    if len(actions) <= 2:
        return 0
    simplified = []
    for action in actions:
        key = {
            'type': action.get('action_type'),
            'target': action.get('target') if action.get('action_type') in ['click', 'long_press'] else None
        }
        simplified.append(json.dumps(key, sort_keys=True))
    
    repeat_count = 0

    for i in range(len(simplified) - 2):
        if simplified[i] == simplified[i+1] == simplified[i+2]:
            repeat_count += 1
            
    for i in range(len(simplified) - 4):
        if (simplified[i] == simplified[i+2] == simplified[i+4] and
            simplified[i+1] == simplified[i+3]):
            repeat_count += 2 
    
    return repeat_count

def count_placeholder_abuse(actions: List[dict]) -> int:
    if len(actions) == 0:
        return 0
    
    current_action = actions[0]
    
    if (current_action.get('action_type') in ['click', 'long_press'] and 
        str(current_action.get('target', '')) == "0"):
        return 1  
    return 0

def calculate_alignment_score(alignments: List[Tuple[int, int, float]], pred_actions: List[dict], gt_actions: List[dict]) -> float:
    total_reward = 0.0
    
    # Calculate reward for matched actions
    for pred_idx, gt_idx, detail_similarity in alignments:
        is_current = (pred_idx == 0)  # First predicted action is "current"
        
        # Apply different weights for current vs future actions
        if is_current:
            type_weight = CURRENT_TYPE_WEIGHT
            detail_weight = CURRENT_DETAIL_WEIGHT
        else:
            type_weight = FUTURE_TYPE_WEIGHT
            detail_weight = FUTURE_DETAIL_WEIGHT
        
        # Type score (binary) + detail score (continuous)
        type_score = type_weight  # Always 1.0 since alignment ensures type match
        detail_score = detail_weight * detail_similarity
        step_reward = type_score + detail_score
        
        # Apply discount factor
        if is_current:
            discounted_reward = step_reward
        else:
            discounted_reward = step_reward * (GAMMA ** pred_idx)
        
        total_reward += discounted_reward

    # Completion bonus - fixed indentation
    if len(pred_actions) > 0 and pred_actions[-1].get('status') == 'done':
        total_reward += 0.1
        
    # Coverage penalty for unmatched actions
    matched_pred = len(alignments)
    matched_gt = len(alignments)
    extra_pred = len(pred_actions) - matched_pred
    missed_gt = len(gt_actions) - matched_gt
    
    coverage_penalty = extra_pred * 0.1 + missed_gt * 0.15  # Missing GT actions penalized more
    
    # Normalize 
    max_possible_reward = 1.0 + 0.1  # Current action + completion bonus
    for i in range(1, len(gt_actions)):
        max_possible_reward += (GAMMA ** i)
    
    if max_possible_reward > 0:
        normalized_reward = (total_reward - coverage_penalty) / max_possible_reward
    else:
        normalized_reward = 0.0
    
    return max(0.0, min(1.0, normalized_reward))
    
    
def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Calculate accuracy reward using sequence alignment and discounted rewards.
    """
    try:
        pred_actions = parse_actions_from_text(predict)
        gt_actions = parse_actions_from_text(ground_truth)
        
        if len(pred_actions) == 0 or len(gt_actions) == 0:
            return 0.0

        max_allowed_steps = len(gt_actions) + 3
        if len(pred_actions) > max_allowed_steps:
            return 0.05
            
        # Find best alignment between sequences
        alignments = find_best_alignment(pred_actions, gt_actions)
        base_score = calculate_alignment_score(alignments, pred_actions, gt_actions)
        repeat_count = count_repetitions(pred_actions)
        repetition_penalty = repeat_count * 0.2

        placeholder_penalty = count_placeholder_abuse(pred_actions) * 0.15
        
        final_score = base_score - repetition_penalty - placeholder_penalty
        return max(0.0, min(1.0, final_score))
        
    except Exception:
        return 0.0

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    """
    Compute final score combining format and accuracy rewards.
    
    Args:
        predicts: List of model prediction texts
        ground_truths: List of ground truth texts
        format_weight: Weight for format score (default: 0.1)
        
    Returns:
        List of dictionaries with 'overall', 'format', and 'accuracy' scores
    """
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        format_score = format_reward(predict)
        accuracy_score = accuracy_reward(predict, ground_truth)
        scores.append(
            {
                "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
