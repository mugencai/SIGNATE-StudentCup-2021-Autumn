# Signate-Student-Cup-2021-Autumn



## コンペの趣旨

オペレーション改善に向けて、より正確に自転車数を予測する機械学習モデルの構築  

  

  
  

## 配布データ

```
┣ data/
    ┣ status.csv            
    ┣ station.csv           
    ┣ weather.csv    		    
    ┣ trip.csv              
```

参加規約より、配布データは各自でコンペ公式ページからダウンロードしてください。

  
  
  
 
## コード構成

```
┣ code.ipynb　　　　　　
┣ code/　　　　　　
    ┣ preprocess.py         
    ┣ train.py      		    
    ┣ predict.py          	
    ┣ main.py      		   	  
    ┣ tool.py   		      	
┣ picture/　　　　　　
    ┣ slide_window.gif      
    ┣ create_gif.ipynb      		
```
  
  


## 注意点

<img src="picture\slide_window.gif" style="zoom:67%;" />

時系列データの場合、Validationのやり方を間違えることによって、異常に高い精度が出てしまうというデータリークは発生しやすいので、交差検証の作り方は要注意です。

時系列データに適用する交差検証について、具体的には2014年9月testを予測する場合、2014年8月のtrainデータで検証，2014年7月以前のtrainデータで訓練；2014年10月testを予測する場合、2014年9月のtrainデータで検証，2014年8月以前のtrainデータで訓練…… 

このように繰り返します。（上の図はイメージ）

  
  
  




  

  
  
## リンク

コンペ公式ページ: https://signate.jp/competitions/550

交差検証の作り方: https://signate.jp/competitions/550/discussions/cv-1
