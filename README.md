# Gensim-as-Service  

## Introduction  

1. 这是一个基于Gensim调用中文词向量的Web Server，提供了几个简要API。

2. [Here](https://github.com/Embedding/Chinese-Word-Vectors)提供了多种中文预训练模型，下载之后解压到data文件夹，替换embedding文件夹中的sample文件即可。

3. 'python Web.py' 即可启用服务，启用服务器前请修改'Web.py'主函数中的地址，改为本机ip。

4. 启用服务以后在浏览器输入以下链接，即可得到对应的json文件，推荐使用chrome，安装JSONView插件，即可直接预览返回的json结果。

## Requirements
python=3.6, gensim, flask 

## API List（使用前请先替换为本地地址）

1. 查看Loading时间，预训练模型名称，预训练模型大小
'http://10.108.17.226:8080/'

2. 查询一个词的embedding  
'http://10.108.17.226:8080/api/embedding?word=中国'

3. 查询两个词的similarity  
'http://10.108.17.226:8080/api/similarity?word1=中国&word2=美国'

4. 查询和Pos_wordlist中词最相似,和Neg_wordlist中词最不相似的n个词  
'http://10.108.17.226:8080/api/topn?n=5&pos=中国%20美国&neg=日本%20俄罗斯'  

   也可以单独查询  
   'http://10.108.17.226:8080/api/topn?n=5&pos=中国%20美国'  

5. 查询两个词集合之间的相似度  
'http://10.108.17.226:8080/api/n_similarity?wlist1=中国%20美国&wlist2=日本'
