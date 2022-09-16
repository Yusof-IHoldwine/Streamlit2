import streamlit as st
from streamlit_option_menu import option_menu

from PIL import Image
import streamlit_authenticator as stauth
import database as db

from PIL import Image

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets
import pytorch_lightning as pl

import torchmetrics
from torchmetrics.functional import accuracy

from torchvision.models import resnet101
    
selected = option_menu(
        menu_title=None,
        options=["Home", "Registration", "Purpose"],
        icons=["house","pc","envelope"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

if selected == "Home":
    #st.title(f"You have selected {selected}")
    # cookie_expiry_daysでクッキーの有効期限を設定可能。認証情報の保持期間を設定でき値を0とするとアクセス毎に認証を要求する
    users = db.fetch_all_users()

    usernames = [user["key"] for user in users]
    names = [user["name"] for user in users]
    hashed_passwords = [user["password"] for user in users]

    authenticator = stauth.Authenticate(names, usernames, hashed_passwords, "sales_dashboard", "abcdef", cookie_expiry_days=30)

        # ログインメソッドで入力フォームを配置
    names, authentication_status, username = authenticator.login('Login', 'main')



    # 返り値、authenticaton_statusの状態で処理を場合分け
    if authentication_status:
        # logoutメソッドでaurhenciationの値をNoneにする
        authenticator.logout('Logout', 'main')
        st.write('Welcome *%s*' % (names))
        st.title('犬猫判断')

        image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

        @st.cache
        def load_image(image_file):
            img = Image.open(image_file)
            return img


        if image_file is not None:
            st.image(load_image(image_file), width=250)

            # ResNet を特徴抽出機として使用
            feature = resnet101(pretrained=True)

            class Net(pl.LightningModule):

                def __init__(self):
                    super().__init__()

                    self.feature = resnet101(pretrained=True)
                    self.fc = nn.Linear(1000, 2)


                def forward(self, x):
                    h = self.feature(x)
                    h = self.fc(h)
                    return h


                def training_step(self, batch, batch_idx):
                    x, t = batch
                    y = self(x)
                    loss = F.cross_entropy(y, t)
                    self.log('train_loss', loss, on_step=True, on_epoch=True)
                    self.log('train_acc', accuracy(y.softmax(dim=-1), t), on_step=True, on_epoch=True)
                    return loss


                def validation_step(self, batch, batch_idx):
                    x, t = batch
                    y = self(x)
                    loss = F.cross_entropy(y, t)
                    self.log('val_loss', loss, on_step=False, on_epoch=True)
                    self.log('val_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
                    return loss


                def test_step(self, batch, batch_idx):
                    x,t = batch
                    y = self(x)
                    loss = F.cross_entropy(y,t)
                    self.log('test_loss', loss, on_step=False, on_epoch=True)
                    self.log('test_acc', accuracy(y.softmax(dim=-1), t), on_step=False, on_epoch=True)
                    return loss


                def configure_optimizers(self):
                    optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
                    return optimizer


            load_path = torch.load('dac.pt',map_location=lambda storage, loc: storage)
            # ネットワークの準備
            net = Net().cpu().eval()
            # 重みの読み込み
            net.load_state_dict(load_path,strict=False)


            # データセットの変換を定義
            transform = transforms.Compose([
                transforms.Resize(size = (224, 224)),
                transforms.ToTensor(),
            ])

            img = Image.open(image_file)

            img = transform(img)

            x = img
            y = net(x.unsqueeze(0))
            y = F.softmax(y,dim=-1)
            y = torch.argmax(y)

            if y==torch.tensor(1):
                st.write("推論結果は犬です")
            else:
                st.write("推論結果は猫です")
        

    elif authentication_status == False:
        st.error('Username/password is incorrect')
    elif authentication_status == None:
        st.warning('Please enter your username and password')



if selected == "Registration":
    #st.title(f"You have selected {selected}")

    new_username = [st.text_input("Username")]
    new_name = [st.text_input("Name")]
    new_password = [st.text_input("Password", type='password')]
    hashed_password = stauth.Hasher(new_password).generate()

    if st.button("Signup"):
        for (username, name, hash_password) in zip(new_username, new_name, hashed_password):
            db.insert_user(username, name, hash_password)
        st.success("You have successfully created an valid account")
        st.info("Go to Login Menu to login")


if selected == "Purpose":
    st.title(f"You have selected {selected}")
    st.write("・Google colabで作成した機能の出力")
    st.write("・Streamlitを用いた記述でログイン機能の実装")
    st.write("・オンライン上に公開")

