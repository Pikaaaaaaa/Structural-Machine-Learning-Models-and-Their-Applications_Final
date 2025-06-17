import re
import matplotlib.pyplot as plt
import numpy as np

faa_log_file = r"C:\Users\user\Desktop\FAA_pth\FAA_test.pth.log"     
ra_log_file = r"C:\Users\user\Desktop\RA_pth\RA_test.pth.log"
save_directory = r"C:\Users\user\Desktop\FAA_pth" 

def parse_training_log(log_file_path, method_name):
    """解析訓練日誌檔案"""
    epochs = []
    train_losses = []
    train_accs = []
    test_losses = []
    test_accs = []
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 使用正則表達式匹配epoch資訊
            match = re.search(r'epoch=(\d+).*?loss=([\d.]+).*?top1=([\d.]+).*?test.*?loss=([\d.]+).*?top1=([\d.]+)', line)
            if match:
                epoch, train_loss, train_acc, test_loss, test_acc = match.groups()
                epochs.append(int(epoch))
                train_losses.append(float(train_loss))
                train_accs.append(float(train_acc))
                test_losses.append(float(test_loss))
                test_accs.append(float(test_acc))
    
    print(f"{method_name} - Total epochs: {len(epochs)}")
    print(f"{method_name} - Final train acc: {train_accs[-1]:.4f}")
    print(f"{method_name} - Final test acc: {test_accs[-1]:.4f}")
    print(f"{method_name} - Final test error: {(1-test_accs[-1])*100:.2f}%")
    
    return epochs, train_losses, train_accs, test_losses, test_accs

def plot_comparison_charts(faa_data, ra_data, save_dir=save_directory):
    """畫六張比較圖並儲存"""
    
    faa_epochs, faa_train_losses, faa_train_accs, faa_test_losses, faa_test_accs = faa_data
    ra_epochs, ra_train_losses, ra_train_accs, ra_test_losses, ra_test_accs = ra_data
    
    # 設置全局字體和樣式
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    
    # 1. Training Loss比較圖
    plt.figure(figsize=(10, 6))
    plt.plot(faa_epochs, faa_train_losses, 'b-', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_train_losses, 'r-', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}training_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training loss comparison saved as: training_loss_comparison.png")
    
    # 2. Testing Loss比較圖
    plt.figure(figsize=(10, 6))
    plt.plot(faa_epochs, faa_test_losses, 'b-', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_test_losses, 'r-', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Testing Loss', fontsize=12)
    plt.title('Testing Loss Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}testing_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Testing loss comparison saved as: testing_loss_comparison.png")
    
    # 3. Training Accuracy比較圖
    plt.figure(figsize=(10, 6))
    faa_train_acc_percent = [acc * 100 for acc in faa_train_accs]
    ra_train_acc_percent = [acc * 100 for acc in ra_train_accs]
    
    plt.plot(faa_epochs, faa_train_acc_percent, 'g-', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_train_acc_percent, 'orange', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Accuracy (%)', fontsize=12)
    plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}training_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training accuracy comparison saved as: training_accuracy_comparison.png")
    
    # 4. Testing Accuracy比較圖
    plt.figure(figsize=(10, 6))
    faa_test_acc_percent = [acc * 100 for acc in faa_test_accs]
    ra_test_acc_percent = [acc * 100 for acc in ra_test_accs]
    
    plt.plot(faa_epochs, faa_test_acc_percent, 'g-', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_test_acc_percent, 'orange', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Testing Accuracy (%)', fontsize=12)
    plt.title('Testing Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}testing_accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Testing accuracy comparison saved as: testing_accuracy_comparison.png")
    
    # 5. Training Error Rate比較圖
    plt.figure(figsize=(10, 6))
    faa_train_error_percent = [(1 - acc) * 100 for acc in faa_train_accs]
    ra_train_error_percent = [(1 - acc) * 100 for acc in ra_train_accs]
    
    plt.plot(faa_epochs, faa_train_error_percent, 'purple', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_train_error_percent, 'brown', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Error Rate (%)', fontsize=12)
    plt.title('Training Error Rate Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}training_error_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Training error rate comparison saved as: training_error_rate_comparison.png")
    
    # 6. Testing Error Rate比較圖
    plt.figure(figsize=(10, 6))
    faa_test_error_percent = [(1 - acc) * 100 for acc in faa_test_accs]
    ra_test_error_percent = [(1 - acc) * 100 for acc in ra_test_accs]
    
    plt.plot(faa_epochs, faa_test_error_percent, 'purple', label='Fast AutoAugment', linewidth=2, marker='o', markersize=3)
    plt.plot(ra_epochs, ra_test_error_percent, 'brown', label='RandAugment', linewidth=2, marker='s', markersize=3)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Testing Error Rate (%)', fontsize=12)
    plt.title('Testing Error Rate Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{save_dir}testing_error_rate_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Testing error rate comparison saved as: testing_error_rate_comparison.png")

def print_comparison_summary(faa_data, ra_data):
    """印出比較摘要"""
    faa_epochs, faa_train_losses, faa_train_accs, faa_test_losses, faa_test_accs = faa_data
    ra_epochs, ra_train_losses, ra_train_accs, ra_test_losses, ra_test_accs = ra_data
    
    print("\n" + "="*60)
    print("FAST AUTOAUGMENT vs RANDAUGMENT COMPARISON")
    print("="*60)
    
    print(f"{'Metric':<25} {'FAA':<15} {'RA':<15} {'Difference':<15}")
    print("-"*60)
    
    # Final accuracies
    faa_final_test_acc = faa_test_accs[-1] * 100
    ra_final_test_acc = ra_test_accs[-1] * 100
    acc_diff = faa_final_test_acc - ra_final_test_acc
    
    print(f"{'Final Test Accuracy':<25} {faa_final_test_acc:.2f}%{'':<8} {ra_final_test_acc:.2f}%{'':<8} {acc_diff:+.2f}%")
    
    # Final error rates
    faa_final_error = (1 - faa_test_accs[-1]) * 100
    ra_final_error = (1 - ra_test_accs[-1]) * 100
    error_diff = faa_final_error - ra_final_error
    
    print(f"{'Final Test Error':<25} {faa_final_error:.2f}%{'':<8} {ra_final_error:.2f}%{'':<8} {error_diff:+.2f}%")
    
    # Final losses
    faa_final_loss = faa_test_losses[-1]
    ra_final_loss = ra_test_losses[-1]
    loss_diff = faa_final_loss - ra_final_loss
    
    print(f"{'Final Test Loss':<25} {faa_final_loss:.4f}{'':<8} {ra_final_loss:.4f}{'':<8} {loss_diff:+.4f}")
    
    print("\n" + "="*60)
    if acc_diff > 0:
        print(f"🏆 Fast AutoAugment achieves {acc_diff:.2f}% higher test accuracy!")
    else:
        print(f"🏆 RandAugment achieves {abs(acc_diff):.2f}% higher test accuracy!")

# 主程式
if __name__ == "__main__":
    import os
    
    # 檔案路徑設定
    faa_log_file = faa_log_file    # Fast AutoAugment log檔案
    ra_log_file = ra_log_file      # RandAugment log檔案
    save_directory = save_directory                 # 圖片儲存路徑
    
    # 檢查檔案是否存在
    print("Current working directory:", os.getcwd())
    print("Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith('.log'):
            print(f"  📄 {file}")
    
    print(f"\nLooking for:")
    print(f"  🔍 {faa_log_file} - {'✅ Found' if os.path.exists(faa_log_file) else '❌ Not found'}")
    print(f"  🔍 {ra_log_file} - {'✅ Found' if os.path.exists(ra_log_file) else '❌ Not found'}")
    
    # 如果檔案不存在，提示使用者修改檔名
    if not os.path.exists(faa_log_file):
        print(f"\n❌ Error: {faa_log_file} not found!")
        print("Please check the filename and update the variable 'faa_log_file' in the code.")
        exit()
    
    if not os.path.exists(ra_log_file):
        print(f"\n❌ Error: {ra_log_file} not found!")
        print("Please check the filename and update the variable 'ra_log_file' in the code.")
        exit()
    
    print("\nParsing training logs...")
    
    # 解析兩個日誌檔案
    faa_data = parse_training_log(faa_log_file, "Fast AutoAugment")
    ra_data = parse_training_log(ra_log_file, "RandAugment")
    
    # 印出比較摘要
    print_comparison_summary(faa_data, ra_data)
    
    # 畫比較圖並儲存
    print("\nGenerating comparison charts...")
    plot_comparison_charts(faa_data, ra_data, save_directory)
    
    print("\n🎉 All comparison charts saved successfully!")
    print("📁 Generated files:")
    print(f"   • {save_directory}training_loss_comparison.png")
    print(f"   • {save_directory}testing_loss_comparison.png")
    print(f"   • {save_directory}training_accuracy_comparison.png")
    print(f"   • {save_directory}testing_accuracy_comparison.png")
    print(f"   • {save_directory}training_error_rate_comparison.png")
    print(f"   • {save_directory}testing_error_rate_comparison.png")