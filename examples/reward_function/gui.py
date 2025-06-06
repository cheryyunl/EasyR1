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
SCREEN_WIDTH = 1080                  # Target screen width
SCREEN_HEIGHT = 2400                 # Target screen height
COORDINATE_TOLERANCE = 0.04          # 4% of max screen dimension
FUTURE_TOLERANCE_MULTIPLIER = 1.5    # Looser tolerance for future actions

# Reward weights: current vs future actions
CURRENT_TYPE_WEIGHT = 0.6    # Current action type importance
CURRENT_DETAIL_WEIGHT = 0.4  # Current action detail importance  
FUTURE_TYPE_WEIGHT = 0.8     # Future action type importance (higher!)
FUTURE_DETAIL_WEIGHT = 0.2   # Future action detail importance (lower!)

def format_reward(predict: str) -> float:
    """
    Check if prediction follows correct Step format.
    Expected: Step X: "screenshot_abstraction": "...", "action": {...}, "status": "..."
    """
    # Find all step patterns (status is optional for backward compatibility)
    step_pattern = r'Step\s+\d+:[^"]*"screenshot_abstraction":[^"]*"[^"]*"[^"]*"action":\s*\{[^}]+\}(?:[^"]*"status":[^"]*"[^"]*")?'
    steps = re.findall(step_pattern, predict, re.DOTALL)
    
    if len(steps) == 0:
        return 0.0
    
    # Validate JSON format for each step
    valid_steps = 0
    for step in steps:
        try:
            action_match = re.search(r'"action":\s*(\{[^}]+\})', step)
            if action_match:
                json.loads(action_match.group(1))
                status_match = re.search(r'"status":\s*"([^"]+)"', step)
                if status_match:
                    status = status_match.group(1)
                    if status in ['done', 'not done']:
                        valid_steps += 1
                else:
                    valid_steps += 1
        except:
            continue
    
    return valid_steps / len(steps)

def parse_actions_from_text(text: str) -> List[dict]:
    """Extract action sequence from text."""
    actions = []
    pattern = r'Step\s+\d+:[^"]*"action":\s*(\{[^}]+\})'
    
    matches = re.findall(pattern, text, re.DOTALL)
    for action_str in matches:
        try:
            actions.append(json.loads(action_str))
        except:
            continue
    
    return actions

def calculate_action_similarity(pred_action: dict, gt_action: dict) -> float:
    """Calculate similarity score between two actions (0-1)."""
    # Type must match
    if pred_action.get('action_type') != gt_action.get('action_type'):
        return 0.0
    
    action_type = pred_action.get('action_type')
    
    # Calculate detail similarity (existing logic)
    detail_similarity = 0.0
    
    if action_type in ['click', 'long_press']:
        # Coordinate similarity
        pred_x, pred_y = pred_action.get('x', 0), pred_action.get('y', 0)
        gt_x, gt_y = gt_action.get('x', 0), gt_action.get('y', 0)
        
        distance = np.sqrt((pred_x - gt_x)**2 + (pred_y - gt_y)**2)
        max_distance = COORDINATE_TOLERANCE * max(SCREEN_WIDTH, SCREEN_HEIGHT) * 2
        
        detail_similarity = max(0, 1 - distance / max_distance)
        
    elif action_type == 'scroll':
        detail_similarity = 1.0 if pred_action.get('direction') == gt_action.get('direction') else 0.0
        
    elif action_type in ['input_text', 'open_app']:
        # Text similarity (substring matching)
        pred_text = str(pred_action.get('text', '') or pred_action.get('app_name', '')).lower().strip()
        gt_text = str(gt_action.get('text', '') or gt_action.get('app_name', '')).lower().strip()
        
        detail_similarity = 1.0 if (pred_text in gt_text) or (gt_text in pred_text) else 0.0
        
    elif action_type in ['navigate_home', 'navigate_back']:
        detail_similarity = 1.0
        
    elif action_type == 'wait':
        pred_time = pred_action.get('seconds', 1)
        gt_time = gt_action.get('seconds', 1)
        time_diff = abs(pred_time - gt_time)
        detail_similarity = 1.0 if time_diff <= 2 else max(0, 1 - time_diff / 10)
    else:
        # Unknown action type, only check status
        detail_similarity = 0.0

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
            similarity = calculate_action_similarity(pred_actions[i], gt_actions[i])
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
                
            similarity = calculate_action_similarity(pred_action, gt_action)
            
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

def accuracy_reward(predict: str, ground_truth: str) -> float:
    """
    Calculate accuracy reward using sequence alignment and discounted rewards.
    """
    try:
        pred_actions = parse_actions_from_text(predict)
        gt_actions = parse_actions_from_text(ground_truth)
        
        if len(pred_actions) == 0 or len(gt_actions) == 0:
            return 0.0
        
        # Find best alignment between sequences
        alignments = find_best_alignment(pred_actions, gt_actions)
        
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
        
        # Coverage penalty for unmatched actions
        matched_pred = len(alignments)
        matched_gt = len(alignments)
        extra_pred = len(pred_actions) - matched_pred
        missed_gt = len(gt_actions) - matched_gt
        
        coverage_penalty = extra_pred * 0.1 + missed_gt * 0.15  # Missing GT actions penalized more
        
        # Normalize to 0-1 range
        max_possible_reward = 1.0  # Current action
        for i in range(1, len(gt_actions)):
            max_possible_reward += (GAMMA ** i)
        
        if max_possible_reward > 0:
            normalized_reward = (total_reward - coverage_penalty) / max_possible_reward
        else:
            normalized_reward = 0.0
        
        return max(0.0, min(1.0, normalized_reward))
        
    except Exception:
        return 0.0

def compute_score(predict: str, ground_truth: str, format_weight: float = 0.5) -> Dict[str, float]:
    """
    Compute final score combining format and accuracy rewards.
    
    Args:
        predict: Model prediction text
        ground_truth: Ground truth text
        format_weight: Weight for format score (default: 0.5)
        
    Returns:
        Dictionary with 'overall', 'format', and 'accuracy' scores
    """
    format_score = format_reward(predict)
    accuracy_score = accuracy_reward(predict, ground_truth)
    
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
