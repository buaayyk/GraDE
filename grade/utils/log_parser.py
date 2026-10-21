import re
import ast
import pathlib
from typing import Dict, Union

def parse_training_log(log_path: Union[str, pathlib.Path]):
    log_path = pathlib.Path(log_path)

    if not log_path.exists():
        raise FileNotFoundError(f"{log_path}")

    epoch_pattern = r'epoch=(\d+)'
    training_loss_pattern = r'Training loss: ([\d.]+)'
    training_loss_x_pattern = r'Training loss X: ([\d.]+)'
    training_loss_e_pattern = r'Training loss E: ([\d.]+)'
    testing_metrics_pattern = r"Testing metrics: (\{[^}]+\})"

    results = {}
    current_epoch = None

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            epoch_match = re.search(epoch_pattern, line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                if current_epoch not in results:
                    results[current_epoch] = {}
            
            training_loss_match = re.search(training_loss_pattern, line)
            if training_loss_match and current_epoch is not None:
                training_loss = float(training_loss_match.group(1))
                results[current_epoch]['Training loss'] = training_loss

            training_loss_x_match = re.search(training_loss_x_pattern, line)
            if training_loss_x_match and current_epoch is not None:
                training_loss_x = float(training_loss_x_match.group(1))
                results[current_epoch]['Training loss X'] = training_loss_x

            training_loss_e_match = re.search(training_loss_e_pattern, line)
            if training_loss_e_match and current_epoch is not None:
                training_loss_e = float(training_loss_e_match.group(1))
                results[current_epoch]['Training loss E'] = training_loss_e
            
            testing_metrics_match = re.search(testing_metrics_pattern, line)
            if testing_metrics_match and current_epoch is not None:
                try:
                    metrics_str = testing_metrics_match.group(1)
                    metrics_dict = ast.literal_eval(metrics_str)
                    
                    if 'X_acc' in metrics_dict:
                        results[current_epoch]['X_acc'] = metrics_dict['X_acc']
                    if 'E_acc' in metrics_dict:
                        results[current_epoch]['E_acc'] = metrics_dict['E_acc']
                        
                except (ValueError, SyntaxError) as e:
                    print(f"parsing Testing metrics fails (epoch {current_epoch}): {e}")
                    continue
        
        return results