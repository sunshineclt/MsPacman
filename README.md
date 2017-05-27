# MsPacman

require moduel:
  tensorflow
  gym
  cv2(from opencv)

1. cut origin shape(210, 160, 3) into (160, 160, 1) through cv2
2. create training network, 3 hidden layers, 2 fully connected layers, with one time max pooling(2x2)
3. train, save results per 100000 timesteps
