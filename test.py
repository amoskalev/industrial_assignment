from models import *
from utils import *

def test_model(model, test_data, Flatten=False):
    """Perform a simple accuracy metric test
    
    Args:
        model: trained NN model that returns (prediction, _)
        test_data: tuple(X_test, y_test)
        Flatten: if True, an input to the network will be flattened. Used for shallow models
        
    Returns:
        print simple accuracy score
    """

    X_test, y_test = test_data[0], test_data[1]
    
    model.train(False) # disable dropout / use averages for batch_norm
    test_batch_acc = []
    
    for X_batch, y_batch in iterate_minibatches(X_test, y_test, 100):
        if Flatten:
            logits,_ = model(Variable(torch.FloatTensor(X_batch).view(torch.FloatTensor(X_batch).size(0),-1)).cuda())
        else:
            logits,_ = model(Variable(torch.FloatTensor(X_batch).cuda()))
        y_pred = logits.cpu().max(1)[1].data.numpy()
        test_batch_acc.append(np.mean(y_batch == y_pred))

    test_accuracy = np.mean(test_batch_acc)

    print("Results:")
    print("  test accuracy:\t\t{:.2f} %".format(
        test_accuracy * 100))