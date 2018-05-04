from utils import *
from models import *

def train_network(model, opt, batch_size=64, num_epochs=11, train_data=None, validation_data=None):
    """train the given NN model
    
    Args:
        model: NN model to train. Should produce (result, _)
        opt: optimizer
        batch_size: batch size, default 64
        num_epochs: number of epoches to train, default 3
        train_data: tuple(X_train, y_train)
        validation_data: tuple(X_val, y_val). If not difined, model is trained with no validational controll
        
    Returns:
        trained_model
    """
    
    train_loss,val_accuracy = [],[]
    
    if not validation_data is None:
        X_val, y_val = validation_data
        
    X_train, y_train = train_data[0], train_data[1]
     
    for epoch in tqdm_notebook(range(num_epochs)):

        start_time = time.time()
        model.train(True)
        for X_batch, y_batch in iterate_minibatches(X_train, y_train, batch_size):
            # train on batch
            loss,_ = compute_loss(model, X_batch, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            train_loss.append(loss.cpu().data.numpy()[0])

        if epoch == 5:
            opt.param_groups[0]['lr'] = opt.param_groups[0]['lr']/10
            
        if not validation_data is None:
            model.train(False) 
            for X_batch, y_batch in iterate_minibatches(X_val, y_val, batch_size):
                logits,_ = model(Variable(torch.FloatTensor(X_batch).cuda()))
                y_pred = logits.max(1)[1].cpu().data.numpy()
                val_accuracy.append(np.mean(y_batch == y_pred))

            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss (in-iteration): \t{:.6f}".format(
                np.mean(train_loss[-len(X_train) // batch_size :])))
            print("  validation accuracy: \t\t\t{:.2f} %".format(
                np.mean(val_accuracy[-len(X_val) // batch_size :]) * 100))
            
    model.train(True)
    return model

def fine_tune(model, opt, batch_size=64, num_epochs=3, train_data=None):
    """Fine tune the given model with provided data
    
    Args:
        model: NN model to fine_tune. Should produce (result, _)
        opt: optimizer
        batch_size: batch size, default 64
        num_epochs: number of epoches to train, default 3
        train_data: tuple(X_train, y_train)
        
    Returns:
        trained_model
    """
    model.train(True)
    X_fine, y_fine = train_data[0], train_data[1]
    
    for epoch in tqdm(range(num_epochs)):
        for X_batch, y_batch in iterate_minibatches(X_fine, y_fine, batch_size):
            loss,_ = compute_loss(model, X_batch, y_batch)
            loss.backward()
            opt.step()
            opt.zero_grad()
            
    return model