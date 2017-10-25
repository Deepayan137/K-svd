import  numpy as np
from numpy import *
from PIL import Image
import matplotlib.pyplot as plt
from numpy import *
import scipy
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit, orthogonal_mp_gram
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
from sklearn.datasets import make_sparse_coded_signal
from pylab import *
from PIL import Image


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- R E A D I N G - I M A G E -- F R O M - F O L D E R---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- I M A G E L O A +D --This module implements the KSVD algorithm of "K-SVD:----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # ONE ----- K S V D - A L G O R I T H M --This module implements the KSVD algorithm of "K-SVD:----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

def img2mat(s):
     im = Image.open(s)
     im.show()
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)
    data = np.reshape(data,(500,500)).astype(float)
#    print data
    return data
#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- O M P  A l g o r i t h m ---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
def omp(A,y):
     r = y 
     indx = []
     j = 0 
     n = A.shape[0] 
     K = A.shape[1]
      x = np.zeros(K).reshape(K,1)
      l = np.nonzero(x)    
       _l = len(l[0])
       S = K/500+2
        
         while _l < S :
             temp = np.fabs(np.dot(A.T, r))
             i = np.argmax(temp)
             indx.append(i)    
             _A = A[:,indx[0:j+1]]
              _x = np.dot(np.linalg.pinv(_A),y)
              r = y - np.dot(_A,_x)    
              l = np.nonzero(_x)
               _l = len(l[0])
               j += 1     
                x[indx] = _x
                 return x            
              
#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- S V D  A l g o r i t h m ---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
def SVD(Y,A,X):
    #    print "\nSVD"
    K = A.shape[1] 
    n =  Y.shape[0] 
     j = 0
     while j < K :
         a = A[:,j] 
         #        print "\na:"
         #        print a
         x = X[j,:]  
         #        print "\nx:" 
         #        print x
         indx = np.nonzero(x) 
         #        print "\nindex:"+str(indx[0]) 
         if (len(indx[0]) == 0):
             #            print "continue"
              j += 1
              continue;
           _x = x[indx] 
           #        print "\n_x:"
           #        print _x
              _X = X[:,indx].reshape(K,indx[0].shape[0])
              _X[j,:] = 0 
              #        print "\n_X:" 
              #        print _X
              _Y = Y[:,indx].reshape(n,indx[0].shape[0]) 
              #        print "\n_Y:"   
              #        print _Y
                      _E = _Y - np.dot(A,_X) 
                      #        print "\n_E:" 
                      #        print _E
                      U,Sigma,VT = np.linalg.svd(_E) 
                      #        print "\nU:"  
                      #        print U
#        print "\nSigma:"     
#        print Sigma
#        print "\nVT:"        
#        print VT

        X[j,indx] = Sigma[0]*VT[0,:]
#        print "\nX:"            
#        print X
        A[:,j] = U[:,0]         
#        print "\nA:"           
#        print A

        j += 1                  
#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # TWO ----- K S V D - A L G O R I T H M --This module implements the KSVD algorithm of "K-SVD:----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
def ksvd(Y,T,maxErr):
    n =  Y.shape[0]    
    N =  Y.shape[1]    
#    K = N/4 + 2      
    K = N * 3        
    
    X = np.zeros(K * N).reshape(K, N) 
#    A = Y[:,:K]   
    
    A = mat(np.random.rand(n,K))   

#    A[:,:N] = Y       
#    A[:,N:2*N] = Y
#    A[:,2*N:K] = Y
    
    print "\nY:"+str(Y) 
#    print "\nA:"+str(A)    

    i = 0       
    while i < T: 
#        print "\niter:"+str(i)    
        X = OMP(A,Y)            
#        print "\nX:"+str(X)    
        
        SVD(Y,A,X)           
        
        E = Y - np.dot(A,X)  
        Enorm = ( np.linalg.norm(E[:,j]) for j in range(0, Y.shape[1]))
        err = max(Enorm)        
        
        print('\nIter: %d, err: %.3f' % (i, err)); 
        if (err < maxErr):      
            break;              

        out = np.dot(A,X)
        X_adj = np.matrix.getH(mat(X))
       # out = dot(A,X_adj)
       # out_im =Image.fromarray(out.astype(np.uint8))
       # out_im.show()

        i += 1

    print "\nX:"  
    print X 
    print "\nA:"   
    print A 

    out = np.dot(A,X)
    out_im =Image.fromarray(out.astype(np.uint8))
    out_im.show()
    X_adj = np.matrix.getH(mat(X))
#    out = dot(A,X_adj)
    print "\nAX:"+str(out)
#    print str(A.shape[0])+str(A.shape[1])
#    print str(X.shape[0])+str(X.shape[1])
#    print str(out.shape[0])+str(out.shape[1])
    out_im =Image.fromarray(out.astype(np.uint8))
    out_im.show()

    from sklearn.feature_extraction.image import extract_patches_2d
    from sklearn.feature_extraction.image import reconstruct_from_patches_2d

#    data = extract_patches_2d(Y, (8,8))
#    data = data.reshape(data.shape[0], -1)
    data = Y
    m = np.mean(data, axis=0)
    data -= m;

    print data

#    code = ompcode(A, data, 2).T;
    code = OMP(A, data);
    patches = np.dot(A,code);

#    patch_size = (8,8)
    patches += m;
#    patches = np.array(patches).reshape(len(data), *patch_size)
#    ret = Y.copy();
    ret = patches
 #   ret[:, 100 // 2:] = reconstruct_from_patches_2d(patches, (100, 100 // 2))
#    plt.figure()
#    plt.imshow(ret, cmap=plt.cm.gray, interpolation='nearest')

#    plt.show();
    
    out_im =Image.fromarray(ret.astype(np.uint8))
    out_im.show()    
#Y = img2mat("new-lena2-100.png") 
Y = img2mat("new-lena3.png")
'''
Y = np.matrix([[0.0,2.0,3.0,4.0],
               [5.0,6.0,7.0,8.0],
               [9.0,1.0,11.0,12.0],
               [13.0,1.0,15.0,16.0],
               [17.0,18.0,19.0,20.0],
               [21.0,22.0,23.0,24.0],
               ])
A = np.matrix([[1,2],  
               [3,4],   
               [5,6],   
               [7,8],   
               [9,10],  
               [11,12], 
               ])
'''
#y = Y[:,0]
#omp(A,y)

#OMP(A,Y)
ksvd(Y,10000,1)

# import scipy
image = cv2.imread(args['image'], 0)

def loadImage():
    
    im = Image.open("lena.png")
    
    im.show() 
    im = im.convert("L") 
    data = im.getdata()
    data = np.matrix(data)
    print data 
    
    data = np.reshape(data,(500,500))
    grey_im = Image.fromarray(data.astype(np.uint8))
    
    grey_im.show()
    grey_im.save('grey-lena.png')
    
   
    data2 = 5 * np.random.randn(500,500) + data    

    for i in range(1000):
        randX=np.random.random_integers(0,data.shape[0]-1)  
        randY=np.random.random_integers(0,data.shape[1]-1)  
        if np.random.random_integers(0,1)==0:  
            data[randX,randY]=255 
  #      else:  
  #          data[randX,randY]=255   
    print data   
    new_im = Image.fromarray(data2.astype(np.uint8))
    
    new_im.show()
    new_im.save('new-lena3.png')

loadImage()
#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 -----C T C -- K S V D - A L G O R I T H M --This module implements the KSVD algorithm of "K-SVD:----------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

halfNoise = True;

def ompcode(D, X, T):
    gram = dot(D.T, D);
    cov = dot(D.T, X.T);
    
    return orthogonal_mp_gram(gram, cov, T, None,);

def ksvd(Y, K, T):
    
    global D, X;
    
    maxIter = 50;
    maxErr = 0.1;
    
    (P, N) = Y.shape;
    D = mat(np.random.rand(P, K));
    Yt = Y.T;
    
    for i in range(K): 
        D[:,i] /= np.linalg.norm(D[:,i])
    J = 0;
    while ( J < maxIter):
            
        X = ompcode(D,Yt,T);
        for i in range(0, K):
    
            usedXi = nonzero(X[i,:])[0];
    
            if (len(usedXi) == 0):
                    continue;     
     
            tmpX = X;
            tmpX[i,:] = 0;
    
            ER = Y[:,usedXi] - dot(D,tmpX[:,usedXi])
            U, s, V = np.linalg.svd(ER)
    
            X[i,usedXi] = s[0]*V[0,:]
            D[:,i] = U[:,0]
        
        E = Y - dot(D,X)
        Enorm = ( np.linalg.norm(E[:,i]) for i in range(0, N) )
        err = max(Enorm)
        
        print('Iter: %d, err: %.3f' % (J, err));
        if (err < maxErr):
            break;
        
        J += 1;
        
    return D;
        
def mkdict():     
    global D, patch_size, face, halfNoise;
    if (halfNoise):
        data = extract_patches_2d(face[:, :width // 2], patch_size, max_patches=5000)
    else:
        data = extract_patches_2d(face, patch_size, max_patches=5000)
    data = data.reshape(data.shape[0], -1)
    data -= np.mean(data, axis=0)
    data /= np.std(data, axis=0);
    D = ksvd(data.T, 100, 2);
    dispdict();
    
def dispdict():
    global D;
    plt.figure()
    for i in range(0, 100):
        plt.subplot(10, 10, i + 1)
        plt.imshow(D[:,i].reshape(patch_size), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show();

def denoise():
    global D, patch_size, face, width, height, patches, data, ret, dico, code;
    data = extract_patches_2d(face[:, width // 2:], patch_size)
    data = data.reshape(data.shape[0], -1)
    m = np.mean(data, axis=0)
    data -= m;

    
    code = ompcode(D, data, 2).T;
    patches = np.dot(code, D.T);
    patches += m;
    patches = np.array(patches).reshape(len(data), *patch_size)
    ret = face.copy();
    ret[:, width // 2:] = reconstruct_from_patches_2d(patches, (height, width // 2))
    plt.figure()
    plt.imshow(ret, cmap=plt.cm.gray, interpolation='nearest')
    plt.show();

y, X, w = make_sparse_coded_signal(n_samples=100,
                                   n_components=8,
                                   n_features=16,
                                   n_nonzero_coefs=2,
                                   random_state=0)

face = mat(scipy.misc.face(gray=True))
face = face / 255.0;
face = face[::2, ::2] + face[1::2, ::2] + face[::2, 1::2] + face[1::2, 1::2]
face /= 4.0
height, width = face.shape;

if halfNoise:
    face[:, width // 2:] += 0.075 * np.random.randn(height, width // 2)
else:
    face += 0.075 * np.random.randn(height, width);
    
plt.figure()
plt.imshow(face, cmap=plt.cm.gray, interpolation='nearest')
plt.show();

patch_size = (8, 8);

mkdict();

denoise();


#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 -----M Y O M P---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#


def img2mat(s):
    
    im = Image.open(s)
    
    im.show()
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data)
    data = np.reshape(data,(500,500))
    print data
    return data

def omp(A,y): 
    
    print "\nA:"+str(A)
    print "\ny:"+str(y)
    
    r = y 
    print "\nr:"+str(r)
    indx = [] 
    j = 0 
    n = A.shape[0] 
    K = A.shape[1] 
    x = np.zeros(K).reshape(K,1) 
    print "\nx"+str(x)
    print "\nn="+str(n)

    while j < n/2:
        print "\nj:"+str(j)+"-----------------------"
        print "\nA.T"+str(A.T)
        temp = np.fabs(np.dot(A.T, r))
        print "\ntemp:"+str(temp)
        i = np.argmax(temp)
        print "\ni:"+str(i)
        indx.append(i)
        print "\nindex[0:j+1]:"+str(indx[0:j+1])
        _A = A[:,indx[0:j+1]]
        print "\n_A:"+str(_A)
        print "\npinv(_A)"+str(np.linalg.pinv(_A))
        _x = np.dot(np.linalg.pinv(_A),y)
        print "\n_x:"+str(_x)
        r = y - np.dot(_A,_x)
        print "\nr:"+str(r)
        j += 1 
    x[indx] = _x
    print "\nx:"+str(x)
    return x

def cs_omp(y,D):    
    L=3
    residual=y  
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((256))
    for j in range(L):  
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)          
        index[j]=pos
        print "\n"+str(index)
        print str(D[:,index>=0])
        my=np.linalg.pinv(D[:,index[0:j+1]])            
        a=np.dot(my,y)      
        print "\na:"+str(a)
        residual=y-np.dot(D[:,index>=0],a)
   # result[index>=0]=a
   # return  result

def OMP(A,Y): 
    N =  Y.shape[1] 
    K =  A.shape[1] 
    X = np.zeros(K * N).reshape(K, N) 
    j = 0
    while j < N:
        x = omp(A,Y[:,j])
        print str(X[:,j])
        X[:,j] = x.reshape(K)
        j += 1
    print str(X)         
     

Y = img2mat("new-lena.png") 

A = np.matrix([[1,2,3,4],
               [5,6,7,8],
               [9,10,11,12],
               [13,14,15,16],
               [17,18,19,20],
               [21,22,23,24],
               ])

'''Y = np.matrix([[1,2],  
               [3,4],   
               [5,6],   
               [7,8],   
               [9,10],  
               [11,12], 
               ])
'''
y = Y[:,0]
#y = np.array([[1],[2],[3],[4],[5],[6]])
#cs_omp(y,A)
#omp(A,y)
#omp(Y.T,y)

OMP(Y,Y)

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- O M P---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#
im = Image.open('new-lena.png')
im.show()
im = im.convert("L")
im = im.getdata()
im = np.matrix(im)
im = np.reshape(im,(500,500))
print im

#im = np.array(Image.open('lena.png')) #256*256


sampleRate=0.7  
Phi=np.random.randn(500*sampleRate,500)

mat_dct_1d=np.zeros((500,500))
v=range(500)
for k in range(0,500):  
    dct_1d=np.cos(np.dot(v,k*3.1415926/500))
    if k>0:
        dct_1d=dct_1d-np.mean(dct_1d)
    mat_dct_1d[:,k]=dct_1d/np.linalg.norm(dct_1d)

img_cs_1d=np.dot(Phi,im)

def cs_omp(y,D):    
#    L=math.floor(3*(y.shape[0])/4)
    L=100
    residual=y  
    index=np.zeros((L),dtype=int)
    for i in range(L):
        index[i]= -1
    result=np.zeros((500))
    for j in range(L): 
        product=np.fabs(np.dot(D.T,residual))
        pos=np.argmax(product)          
        index[j]=pos
        my=np.linalg.pinv(D[:,index>=0])            
        a=np.dot(my,y)      
        residual=y-np.dot(D[:,index>=0],a)
    result[index>=0]=a
    return  result


sparse_rec_1d=np.zeros((500,500))  
Theta_1d=np.dot(Phi,mat_dct_1d)   
for i in range(500):
    
    column_rec=cs_omp(img_cs_1d[:,i],Theta_1d)
    sparse_rec_1d[:,i]=column_rec;        
img_rec=np.dot(mat_dct_1d,sparse_rec_1d)      

image2=Image.fromarray(img_rec)
image2.show()

#-------------------------------------------------------------------------------------------------------------------#
#------------------------------------------- # 1 ----- S I G M A---------------------------------------------#
#-------------------------------------------------------------------------------------------------------------------#

dataMat = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

U,Sigma,VT = linalg.svd(dataMat)

print "U"
print U

print "\nSigma"
print Sigma

print "\nVT"
print VT