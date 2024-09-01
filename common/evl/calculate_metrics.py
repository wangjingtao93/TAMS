import numpy as np
import cv2

# 输入（单通道预测图，单通道预测图，评价类型）
# 注意，预测图与原图保持，通道，大小一致
# 参考参数（字符串）：iou,dice_coefficient,accuracy,precision,recall,sensitivity,f1,specificity
def calculate_metrics(predict_image, gt_image, evaluate):
    # 将图像转换为二进制数组
    predict_image = np.array(predict_image, dtype=bool)
    gt_image = np.array(gt_image, dtype=bool)

    # 计算True Positive（TP）
    tp = np.sum(np.logical_and(predict_image, gt_image))

    # 计算True Negative（TN）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))

    # 计算False Positive（FP）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))

    # 计算False Negative（FN）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))


    # 计算IOU（Intersection over Union）
    iou = tp / (tp + fn + fp + 1e-7)

    # 计算Dice Coefficient（Dice系数）
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)

    # 计算Accuracy（准确率）
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)

    # 计算precision（精确率）
    precision = tp / (tp + fp + 1e-7)

    # 计算recall（召回率）
    recall = tp / (tp + fn + 1e-7)

    # 计算Sensitivity（敏感度）
    sensitivity = tp / (tp + fn + 1e-7)

    # 计算F1-score
    f1 = 2*(precision*recall)/(precision+recall + 1e-7)

    # 计算Specificity（特异度）
    specificity = tn / (tn + fp + 1e-7)


    if evaluate == "iou": 
        return iou
    
    if evaluate == "dice_coefficient": 
        return dice_coefficient
    
    if evaluate == "accuracy": 
        return accuracy
    
    if evaluate == "precision": 
        return precision
    
    if evaluate == "recall": 
        return recall
    
    if evaluate == "sensitivity": 
        return sensitivity
    
    if evaluate == "f1": 
        return f1
    
    if evaluate == "specificity": 
        return specificity
