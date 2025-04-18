import torch
import copy
import json
from tqdm import tqdm 


def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']


def loss_batch(loss_f1,loss_f2,malis_params, output, target, opt=None):
    loss1 , met1 , met2 = loss_f2(output,target,malis_params[0],malis_params[1])     
    loss = float(10.0)*loss_f1(output.float(), target.float()) + loss1
    #loss = loss_f2(output,target,malis_params[0],malis_params[1])
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item() , met1.item() , met2.item()


def loss_epoch(model,loss_f1,loss_f2,malis_params,dataset_dl,sanity_check=False,opt=None):
    running_loss=0.0
    #print(dataset_dl)
    len_data=len(dataset_dl.dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    iterator = tqdm(dataset_dl)
    for xb, yb in iterator :
        xb=xb.to(device)
        yb=yb.to(device)
        output=model(xb)
        loss_b ,met1_b , met2_b =loss_batch(loss_f1,loss_f2,malis_params, output, yb, opt)
        running_loss += loss_b 
        iterator.set_postfix(train_loss = loss_b , conn_loss = met1_b , disconn_loss = met2_b)

        if sanity_check is True:
            break
    
    loss=running_loss/float(len_data)
    
    return loss 


def train_val(model, params):
    num_epochs=params["num_epochs"]
    loss_func1=params["loss_func1"]
    loss_func2=params["loss_func2"]
    opt=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    test_dl=params["test_dl"]
    sanity_check=params["sanity_check"]
    path2weights=params["path2weights"]
    path2losshist=params["path2losshist"]
    malis_params=params["malis_params"]
    
    #print(params)

    loss_history={
        "train": [],
        "val": []}
    
    
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss=float('inf')    
    
    best_dice=float(0)
    scheduler =  torch.optim.lr_scheduler.ExponentialLR(opt , gamma = 0.95 , verbose = True)
    for epoch in tqdm(range(num_epochs) , total = num_epochs):
        #current_lr=get_lr(opt)
        #print('current lr={}'.format(current_lr))   

        model.train()
        train_loss = loss_epoch(model,loss_func1,loss_func2,malis_params,train_dl,sanity_check,opt)

        loss_history["train"].append(train_loss)
        
        model.eval()
        with torch.no_grad():
            val_loss = loss_epoch(model,loss_func1,loss_func2,malis_params,val_dl,sanity_check)
       
        loss_history["val"].append(val_loss)
        
        if val_loss < best_loss:
            best_loss = copy.deepcopy(val_loss)
            best_model_wts = copy.deepcopy(model.state_dict())            
            torch.save(model.state_dict(), path2weights)
            print("Copied best model weights!")          
            
        print("train loss: %.6f" \
              %(train_loss))
        print("val loss: %.6f" \
              %(val_loss))
        print("-"*10) 
        
       
        with open(path2losshist,"w") as fp:
            json.dump(loss_history, fp)
        scheduler.step()    

    torch.save(model.state_dict(), path2weights[:-3]+"_last.pt")
    print("Copied last model weights!") 

    model.eval()
    with torch.no_grad():
        test_loss = loss_epoch(model,loss_func1,loss_func2,malis_params,test_dl,sanity_check)
        print("Test loss: {}".format(test_loss))
    model.load_state_dict(best_model_wts)
    
    return model, loss_history , test_loss