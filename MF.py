import  torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
class MF(nn.Module):
    def __init__(self,user_nums,item_nums,embedding_dim=64,opt="Adam",reg_user=1e-5,reg_item=1e-5):

        super(MF,self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.items_emb=nn.Embedding(item_nums+1,embedding_dim,device=self.device)
        self.users_emb=nn.Embedding(user_nums+1,embedding_dim,device=self.device)
        self.reg_user=reg_user
        self.reg_item= reg_item
        self.loss_fn=F.cross_entropy
        self.total_loss=0
        self.opt=opt

    def forward(self,user,item):
        user_emb = self.users_emb(user)
        item_emb = self.items_emb(item)
        res =torch.dot(user_emb, item_emb)
        return res


    def fit(self, train_loader):
        criterion = self.loss_fn
        model = self.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        if self.opt == "SGD":
            optimizer = optim.SGD(model.parameters(), lr=0.001)
        elif self.opt == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0,
                                                momentum=0, centered=False)

        for batch_idx ,(user,item,target) in enumerate( train_loader):
            # user,item, target = train_loader[batch_idx][0], train_loader[batch_idx][1], train_loader[batch_idx][2]
            # print(user)
            user=torch.tensor(user,dtype=torch.int32)
            # print(user)
            item=torch.tensor(item,dtype=torch.int32)


            # target.squeeze()
            optimizer.zero_grad()
            output = model(user,item)
            target = torch.tensor([target])
            output = torch.tensor([output])
            loss = criterion(output,target)
            # 加上正则项
            loss = loss + self.reg_user * torch.norm( self.users_emb( user),p=1) +self.reg_item* torch.norm(self.items_emb( item),p=1)
            loss.backward()
            optimizer.step()

    def test(self, test_loader):
        criterion = nn.MSELoss()
        model = self.eval()
        test_MSELoss = 0.0
        test_MAEloss =0.0
        correct = 0

        for batch_idx ,(user,item,target) in enumerate( test_loader):
            user=torch.tensor(user,dtype=torch.int32)
            item=torch.tensor(item,dtype=torch.int32)
            output = model(user, item)
            test_MAEloss +=abs(output-target)
            target=torch.tensor(target, dtype=float)
            # output = torch.tensor(output,dtype=float,requires_grad=True)
            loss = criterion(output,target)
            test_MSELoss +=loss.item()


        test_MSELoss /= len(test_loader)
        test_MAEloss /= len(test_loader)
        print( f'\nTest set: MSE loss: {test_MAEloss:.4f},  RMSELoss:{test_MAEloss**0.5:.4f}, MAELoss:{test_MAEloss:.4f}')

