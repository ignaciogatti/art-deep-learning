import numpy as np
import matplotlib.pyplot as plt


#Auxiliar function to add noise to an image  
def noisy_images(X):
    
    X_noisy = X + 0.5 * np.random.normal(size=X.shape, scale=0.5, loc=0.5)
    return X_noisy


#Auxiliar function to save images
def sample_images(epoch, gen_size, generator):
    r, c = 2, 2
    size = (r*c,) + gen_size
    noise = np.random.normal(0, 1, size= size)
    gen_imgs = generator.predict(noise)
    
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/%d.png" % epoch)
    
    plt.close()
    


#Auxiliar function to plot probas distribution D(x) and D(G(z))
def sample_probas(X, batch_size, gen_size, discriminator, generator):
    plt.title('Generated vs real data')
    
    # Select a random batch of images
    idx = np.random.randint(0, X.shape[0], batch_size)
    imgs = X[idx]
    plt.hist(discriminator.predict(imgs)[:,0],
             label='D(x)', alpha=0.5,range=[0,1])
    
    #Generate random input
    noise = np.random.normal(0, 1, size=gen_size)
    plt.hist(discriminator.predict(generator.predict(noise))[:,0],
             label='D(G(z))',alpha=0.5,range=[0,1])
    plt.legend(loc='best')
    plt.show()