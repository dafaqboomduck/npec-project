import tensorflow as tf
import keras
import keras.ops as ops

@keras.saving.register_keras_serializable()
def dice_bce_loss(y_true, y_pred):
    """
    Custom loss function for unbalanced class distribution.
    """
    # 1. Define weights
    dice_weight = 0.4
    bce_weight = 0.6

    # 2. Calculate Dice Loss
    # smooth = 1e-6
    # y_true_f = K.flatten(K.cast(y_true, "float32"))
    # y_pred_f = K.flatten(y_pred)
    # intersection = K.sum(y_true_f * y_pred_f)

    # dice_score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    # dice_loss_val = 1 - dice_score

    dice_loss_val =  tf.keras.losses.dice(y_true, y_pred)

    # 3. Calculate Binary Cross Entropy
    bce_loss_val = tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # 4. Return weighted sum
    return(dice_weight * dice_loss_val) + (bce_weight * bce_loss_val)

@keras.saving.register_keras_serializable()
def f1(y_true, y_pred):
    """
    F1 Score metric for binary segmentation.
    """
    eps = 1e-8
    def recall_m(y_true, y_pred):
        TP = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
        Positives = ops.sum(ops.round(ops.clip(y_true, 0, 1)))
        recall = TP / (Positives + eps)
        return recall
   
    def precision_m(y_true, y_pred):
        TP = ops.sum(ops.round(ops.clip(y_true * y_pred, 0, 1)))
        Pred_Positives = ops.sum(ops.round(ops.clip(y_pred, 0, 1)))
        precision = TP / (Pred_Positives + eps)
        return precision
   
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
   
    return 2 * ((precision * recall) / (precision + recall + eps))

@keras.saving.register_keras_serializable()
def iou(y_true, y_pred):
    """
    IOU metric for binary segmentation.
    """
    eps = 1e-8
    
    y_true_f = ops.ravel(ops.cast(y_true, "float32"))
    y_pred_f = ops.ravel(y_pred)
    intersection = ops.sum(y_true_f * y_pred_f)

    union = ops.sum(y_true_f) + ops.sum(y_pred_f) - intersection

    iou = intersection/(union + eps)

    return iou