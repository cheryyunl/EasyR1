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
from typing import Dict, List, Tuple, Any

# Configuration
GAMMA = 0.8  # Discount factor for future actions

def format_reward(predict: str) -> float:
    """Check if prediction follows correct format with action_type and step_instruction."""
    if not re.fullmatch(r"<think>.*?</think>\s*<answer>.*?</answer>", predict, re.DOTALL):
        return 0.0
    
    answer_match = re.search(r"<answer>(.*?)</answer>", predict, re.DOTALL)
    if not answer_match:  
        return 0.0
    
    answer_content = answer_match.group(1).strip()
    all_steps = re.findall(r'Step\s+\d+:', answer_content)
    if not all_steps: 
        return 0.0
    
    valid_steps = 0
    for step_match in re.finditer(r'(Step\s+\d+:.*?)(?=Step\s+\d+:|$)', answer_content, re.DOTALL):
        step_text = step_match.group(1).strip()

        # Check required components
        has_screenshot = re.search(r'"screenshot_abstraction":\s*"[^"]*"', step_text)
        has_action = re.search(r'"action":\s*\{[^}]+\}', step_text)
        has_status = re.search(r'"status":\s*"(done|not done)"', step_text)
        
        # Check action contains action_type and step_instruction
        action_valid = False
        if has_action:
            action_match = re.search(r'"action":\s*(\{[^}]+\})', step_text)
            if action_match:
                try:
                    action_dict = json.loads(action_match.group(1))
                    if "action_type" in action_dict and "step_instruction" in action_dict:
                        action_valid = True
                except:
                    pass
        
        if has_screenshot and action_valid and has_status:
            valid_steps += 1
    
    return valid_steps / len(all_steps)

def parse_actions_from_text(text: str) -> List[dict]:
    """Extract action sequence from text, focusing only on action_type and status."""
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
                # Only keep action_type for comparison
                simplified_action = {
                    'action_type': action_dict.get('action_type'),
                    'status': status_match.group(1)
                }
                actions.append(simplified_action)
        except:
            continue
    return actions

def calculate_action_type_similarity(pred_action: dict, gt_action: dict) -> float:
    """Calculate similarity score based only on action_type (0 or 1)."""
    if pred_action.get('action_type') == gt_action.get('action_type'):
        return 1.0
    else:
        return 0.0

def find_best_alignment(pred_actions: List[dict], gt_actions: List[dict]) -> List[Tuple[int, int, float]]:
    """
    Find best action alignment using greedy matching based only on action_type.
    Returns: [(pred_idx, gt_idx, similarity_score), ...]
    """
    # If same length, use position-based matching
    if len(pred_actions) == len(gt_actions):
        alignments = []
        for i in range(len(pred_actions)):
            similarity = calculate_action_type_similarity(pred_actions[i], gt_actions[i])
            alignments.append((i, i, similarity))
        return alignments
    
    # Different lengths: greedy matching with position preference
    alignments = []
    used_gt_indices = set()
    
    for pred_idx, pred_action in enumerate(pred_actions):
        best_gt_idx = None
        best_score = 0.0
        
        for gt_idx, gt_action in enumerate(gt_actions):
            if gt_idx in used_gt_indices:
                continue
                
            similarity = calculate_action_type_similarity(pred_action, gt_action)
            
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
    """Count repetitive action patterns."""
    if len(actions) <= 2:
        return 0
    
    # Simplify to just action types
    action_types = [action.get('action_type') for action in actions]
    
    repeat_count = 0
    
    # Count 3+ consecutive identical actions
    for i in range(len(action_types) - 2):
        if action_types[i] == action_types[i+1] == action_types[i+2]:
            repeat_count += 1
    
    return repeat_count

def calculate_alignment_score(alignments: List[Tuple[int, int, float]], pred_actions: List[dict], gt_actions: List[dict]) -> float:
    """Calculate reward score with gamma discounting."""
    total_reward = 0.0
    
    # Calculate reward for matched actions
    for pred_idx, gt_idx, type_similarity in alignments:
        # Apply discount factor based on step position
        discounted_reward = type_similarity * (GAMMA ** pred_idx)
        total_reward += discounted_reward

    # Completion bonus
    if len(pred_actions) > 0 and pred_actions[-1].get('status') == 'done':
        total_reward += 0.2
        
    # Coverage penalty for unmatched actions
    matched_pred = len(alignments)
    extra_pred = len(pred_actions) - matched_pred
    missed_gt = len(gt_actions) - len(alignments)
    
    coverage_penalty = extra_pred * 0.15 + missed_gt * 0.15
    
    # Normalize by maximum possible reward
    max_possible_reward = sum(GAMMA ** i for i in range(len(gt_actions))) + 0.2
    
    if max_possible_reward > 0:
        normalized_reward = (total_reward - coverage_penalty) / max_possible_reward
    else:
        normalized_reward = 0.0
    
    return max(0.0, min(1.0, normalized_reward))

def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Calculate accuracy reward using simplified action type matching.
    """
    try:
        pred_actions = parse_actions_from_text(predict)
        gt_actions = parse_actions_from_text(ground_truth)
        
        if len(pred_actions) == 0 or len(gt_actions) == 0:
            return 0.0

        # Penalty for excessively long sequences
        max_allowed_steps = len(gt_actions) + 3
        if len(pred_actions) > max_allowed_steps:
            return 0.05
            
        # Find best alignment between sequences
        alignments = find_best_alignment(pred_actions, gt_actions)
        base_score = calculate_alignment_score(alignments, pred_actions, gt_actions)
        
        # Apply repetition penalty
        repeat_count = count_repetitions(pred_actions)
        repetition_penalty = repeat_count * 0.1
        
        final_score = base_score - repetition_penalty 
        return max(0.0, min(1.0, final_score))
        
    except Exception:
        return 0.0

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  
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