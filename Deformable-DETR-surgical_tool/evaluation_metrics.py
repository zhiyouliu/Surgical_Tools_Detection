from armory.utils.metrics import (object_detection_AP_per_class, 
                                  object_detection_true_positive_rate, 
                                  object_detection_misclassification_rate, 
                                  object_detection_disappearance_rate, 
                                  object_detection_hallucinations_per_image
                                )

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class EvaluationMetrics(object):
    '''
        This class provides the evaluation and records for object detection task
    '''
    def __init__(self):
        self.record = {
            "ap": None,
            "dr": AverageMeter(),
            "hpi": AverageMeter(), 
            "mr": AverageMeter(), 
            "tpr": AverageMeter(),
        }
        self.evaluation_metrics = {"ap": object_detection_AP_per_class, 
                                "dr": object_detection_disappearance_rate, 
                                "hpi": object_detection_hallucinations_per_image, 
                                "mr": object_detection_misclassification_rate, 
                                "tpr": object_detection_true_positive_rate,
                                }
        self.all_labels = []
        self.all_preds = []
    
    def update_record(self, metric, result, n=1):
        '''
            Input:
                metric: support of ["ap", "dr", "hpi", "mr", "tpr"]
                result: only for "ap", it is a list of two dictionaries
                        all other metrics, are type of list with only one element
                n: number of batches
        '''
        if metric == 'ap':
            curr_labels = result[0]
            curr_preds = result[1]
            self.all_labels.append(curr_labels)
            self.all_preds.append(curr_preds)
            
        else:
            self.record[metric].update(result[0], n)
    
    def update_eval(self, y_obj, y_pred):
        '''
            Input:
                y_obj: type of list with one dictionary in it
                y_pred: type of list with one dictionary in it,
                         and requires {'labels', 'boxes', 'scores'}
        '''
        n = len(y_obj)
        for metric in self.evaluation_metrics:
            if metric != 'ap':
                result = self.evaluation_metrics[metric](y_obj, self.cast_cuda_to_cpu(y_pred))
                self.update_record(metric, result, n)
            else:
                self.update_record(metric, [y_obj[0], self.cast_cuda_to_cpu(y_pred)[0]])
    
    def access_record(self, metric):
        if metric == 'ap':
            return self.evaluation_metrics[metric](self.all_labels, self.all_preds)
        else:
            return self.record[metric].avg
    
    def access_record_all(self):
        output = {}
        for metric in self.record:
            output[metric] = self.access_record(metric)
        return output

    def cast_cuda_to_cpu(self, y):
        '''
            y: type of list with one dictionary in it,
                and requires {'labels', 'boxes', 'scores'}
        '''
        for  i in range(len(y)):
            for key in y[i]:
                y[i][key] = y[i][key].cpu()
        
        return y


