import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import datasets
#bar graphs
"""
bar_width = 0.50
num_bins = 5
bar1 = np.random.randint(0,100,num_bins)
bar2 = np.random.randint(0,100,num_bins)
#bar3 = np.random.randint(0,100,num_bins)

indices = np.arange(num_bins)

plt.bar(indices,bar1,bar_width,color='b',label='Prof1')
plt.bar(indices+bar_width,bar2,bar_width,color='g',label='prof2')
#plt.bar(indices+2*bar_width,bar2,bar_width,color='r',label='prof3')

plt.xlabel('Final Grade')
plt.ylabel('Frequency')


plt.legend()
plt.xticks(indices+bar_width/2,('A','B','C','D','E'))
"""
#sub-plots
"""
x = np.linspace(0,10)
speed_plot = plt.subplot(2,1,1)
plt.plot(x,np.sin(x),'-',label='sin')
plt.ylabel('speed(m/s)')
plt.setp(speed_plot.get_xticklabels(),visible=False)
plt.grid(True)


plt.subplot(2,1,2,sharex=speed_plot)
plt.plot(x,np.cos(x),'-',label='cos')
plt.ylabel('acc (m/s/s)')
plt.ylabel('time(s)')
plt.grid(True)
"""

#histograms
"""
mu = 0
sigma = 1
vals = mu + sigma * np.random.randn(1000)
plt.hist(vals,50)
plt.xlabel('Bins')
plt.ylabel('Freq')
plt.title('Normal Distr')
plt.grid(True)
"""
#pie charts
"""
labels=['gas','books','rent','car','transport']
values=[100,200,350,500,20]
#to take out the slice from chart
#use explode=explode
explode = (0,0,1,0,1)
#autopct to draw the values on the pie chart
#shadow for shadow effect
plt.pie(values,labels=labels,radius=1,shadow=True,autopct='%f%%')
"""
#boxplot
"""
x = np.random.rand(100)*100
plt.boxplot(x,vert=False)
"""
#line plot
"""
#'b--' line format letter stands for the color
x = np.linspace(0,10)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,'g--',label='sine')
plt.plot(x,y2,'b-',label='cosine')
plt.legend()
plt.grid(True)
"""
#scatter plot
#iris dataset
"""
iris = datasets.load_iris()
x=iris.data[:,1]
y=iris.data[:,2]
z=iris.data[:,0]
colors=iris.target
plt.scatter(x,y,z,c=colors)
plt.grid(True)
"""

#quiver plot and stream plot
#creating a grid of x and y from -10 and +10
"""
X,Y=np.meshgrid(np.arange(-10,10),np.arange(-10,10))
U=-Y
V=X
plt.streamplot(X,Y,U,V)
plt.quiver(X,Y,U,V)
"""

#3D line plot
#eg tracking plane
"""
phi = np.linspace(-6*np.pi,6*np.pi,100)
z=np.linspace(-4,4,100)
x=np.sin(phi)
y=np.cos(phi)

fig=plt.figure()
axes=fig.gca(projection='3d')
axes.plot(x,y,z)
"""

#3D sureface plot
X,Y=np.meshgrid(np.arange(-10,100),np.arange(-10,100))
Z= X**2 + Y**2
g_X,g_X=np.gradient(Z)
g=np.sqrt(g_X**2,g_X**2)
g=g/g.max()
fig=plt.figure()
axes=fig.gca(projection='3d')
axes.plot_surface(X,Y,Z,facecolors=cm.inferno(g))


plt.show()
