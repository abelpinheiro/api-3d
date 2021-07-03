from django.shortcuts import render

# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
from PIL import Image

@csrf_exempt
def reconstruct(request):
    # Inicializa o retorno da requisição
  data = {"success": False}

  # Checa se a requisição é um POST
  if request.method == "POST":

    # Checa se foi enviado para a requisição uma imagem
    if request.FILES.get("image_left", None) is not None:
      image_left = _grab_image(stream = request.FILES["image_left"]) # Pegando a imagem
      image_right = _grab_image(stream = request.FILES["image_right"])
      
      # image = _grab_image(stream = request.FILES["image"]) # AQUI É PARA A OUTRA IMAGEM NO BODY
      
    # Se imagem não tiver sido upada, verificar pela URL
    else:
      url = request.POST.get("url", None)
      if url is None:
        data["error"] = "URL não foi dada"
        return JsonResponse(data)
      disparity = _grab_image(url=url)

    # COMEÇA A PUTARIA DO ALGORITMO
    dist, K, ret, FocalLength, rvecs, tvecs =  getCalibrationData()
    
    image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2GRAY)
    image_right = cv2.cvtColor(image_right, cv2.COLOR_BGR2GRAY)
    
    # Pegar altura e largura. Precisa ser o mesmo tamanho
    w, h = np.shape(image_left)
    
    # Obter matriz de camera otimizada para melhor distorção 
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))
    
    # distorcer as imagens
    img_1_undistorted = cv2.undistort(image_left, K, dist, None, new_camera_matrix)
    img_2_undistorted = cv2.undistort(image_right, K, dist, None, new_camera_matrix)
    
    # Parametros de disparidade
    win_size = 7
    min_disp = -1
    max_disp = 63 #min_disp * 9
    num_disp = max_disp - min_disp # Needs to be divisible by 16

    # Cria objeto com o algoritmo de Matching Block. 
    stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
      numDisparities = num_disp,
      blockSize = 5,
      uniquenessRatio = 5,
      speckleWindowSize = 5,
      speckleRange = 5,
      disp12MaxDiff = 1,
      P1 = 8*3*win_size**2,#8*3*win_size**2,
      P2 =32*3*win_size**2) #32*3*win_size**2)

    # Computando o mapa de disparidade
    disparity_map = stereo.compute(img_1_undistorted, img_2_undistorted)
    
    points = reconstruct3dImage(disparity_map, None) #verificar se funciona enviando apenas o mapa de disparidade. NÃO TA CONSTRUINDO O OBJETO PLY
    #Get color points
    colors = cv2.cvtColor(img_1_undistorted, cv2.COLOR_BGR2RGB)#Get rid of points with value 0 (i.e no depth)
    mask_map = disparity_map > disparity_map.min()#Mask colors and points. 
    output_points = points[mask_map]
    output_colors = colors[mask_map]#Define name for output file
    output_file = 'reconstructed-caixa-02.ply'#Generate point cloud 
    print ("\n Creating the output file... \n")
    create_output(output_points, output_colors, output_file)
    #data["points"] = points

    data["success"] = True
  return JsonResponse(data)
 
def getCalibrationData():
  os.chdir(r'./scene_reconstruction/data')
  dist = np.load('dist.npy')
  K = np.load('K.npy')
  ret = np.load('ret.npy')
  FocalLength = np.load('FocalLength.npy', allow_pickle=True)
  rvecs = np.load('rvecs.npy')
  tvecs = np.load('tvecs.npy')
  return dist, K, ret, FocalLength, rvecs, rvecs

def reconstruct3dImage(disparity, img_1_downsampled):
  focal_length = 3.14
  Q2 = np.float32([[1,0,0,0],
    [0,-1,0,0],
    [0,0,focal_length*0.05,0], #Focal length multiplication obtained experimentally. 
    [0,0,0,1]])#Reproject points into 3D
  points_3D = cv2.reprojectImageTo3D(disparity, Q2)
  #colors = cv2.cvtColor(img_1_downsampled, cv2.COLOR_BGR2RGB)#Get rid of points with value 0 (i.e no depth)
  #mask_map = disparity_map > disparity_map.min()#Mask colors and points. 
  #output_points = points_3D[mask_map]
  #output_colors = colors[mask_map]#Define name for output file
  #output_file = 'reconstructed-caixa-02.ply'#Generate point cloud 
  #print ("\n Creating the output file... \n")
  #create_output(output_points, output_colors, output_file)
  return points_3D

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')

def _grab_image(path = None, stream = None, url = None):
  # if the path is not None, then load the image from disk
  if path is not None:
    image = cv2.imread(path)
    
  # otherwise, the image does not reside on disk
  else:
    # if the URL is not None, then download the image
    if url is not None:
      resp = urllib.urlopen(url)
      data = resp.read()
      
    # if the stream is not None, then the image has been uploaded
    elif stream is not None:
      data = stream.read()
    
    # convert the image to a NumPy array and then read it into
		# OpenCV format
    image = np.asarray(bytearray(data), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

  # return the image
  return image