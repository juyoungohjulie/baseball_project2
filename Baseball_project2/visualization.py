## Original Code
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import logging
import numpy as np


def plot_confusion_matrix(y_test, y_pred, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(20, 16))
    
    norm_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, 
                annot=True,      
                fmt=',d',        
                cmap='Blues',    
                xticklabels=sorted(set(y_test)),
                yticklabels=sorted(set(y_test)),
                annot_kws={
                    'size': 10,
                    'weight': 'bold'
                },
                square=True)
    
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('Actual Label', fontsize=14)
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    
    plt.xticks(fontsize=12, rotation=45, ha='right')
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    plt.gcf().set_size_inches(20, 16)
    
    plt.savefig(save_path, 
                dpi=300, 
                bbox_inches='tight', 
                pad_inches=0.5,
                facecolor='white',  
                edgecolor='none')   
    
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()

def plot_training_history(history, model_name='model', save_dir='results'):

    import os
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Training history plot saved to: {save_path}")
