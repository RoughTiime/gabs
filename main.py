import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import functools
from numbers import Integral
from scipy.sparse import issparse
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import _safe_indexing, check_random_state, check_X_y
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    validate_params,
)

from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances, pairwise_distances_chunked
import random
import csv
import math
import statistics as stat
from streamlit_option_menu import option_menu
from matplotlib.backends.backend_agg import RendererAgg
import io
import seaborn as sns
from PIL import Image
import streamlit.components.v1 as components
import base64
from io import BytesIO
import time
from fpdf import FPDF
import streamlit_ext as ste

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

def is_valid_number(value):
    try:
        int(value)
        return True
    except ValueError:
        return False

def based_on_radius(e):
    return e[2]
def silhouette_samples(X, labels, *, metric="euclidean", **kwds):
      X, labels = check_X_y(X, labels, accept_sparse=["csr"])

      # Check for non-zero diagonal entries in precomputed distance matrix
      if metric == "precomputed":
          error_msg = ValueError(
              "The precomputed distance matrix contains non-zero "
              "elements on the diagonal. Use np.fill_diagonal(X, 0)."
          )
          if X.dtype.kind == "f":
              atol = np.finfo(X.dtype).eps * 100
              if np.any(np.abs(X.diagonal()) > atol):
                  raise error_msg
          elif np.any(X.diagonal() != 0):  # integral dtype
              raise error_msg

      le = LabelEncoder()
      labels = le.fit_transform(labels)
      n_samples = len(labels)
      label_freqs = np.bincount(labels)
      check_number_of_labels(len(le.classes_), n_samples)

      kwds["metric"] = metric
      reduce_func = functools.partial(
          _silhouette_reduce, labels=labels, label_freqs=label_freqs
      )
      results = zip(*pairwise_distances_chunked(X,
                    reduce_func=reduce_func, **kwds))
      intra_clust_dists, inter_clust_dists = results
      intra_clust_dists = np.concatenate(intra_clust_dists)
      inter_clust_dists = np.concatenate(inter_clust_dists)

      sil_samples = inter_clust_dists - intra_clust_dists
      with np.errstate(divide="ignore", invalid="ignore"):
          sil_samples /= np.maximum(intra_clust_dists, inter_clust_dists)
      # nan values are for clusters of size 1, and should be 0
      return np.nan_to_num(sil_samples)
def _silhouette_reduce(D_chunk, start, labels, label_freqs):
    n_chunk_samples = D_chunk.shape[0]
    # accumulate distances from each sample to each cluster
    cluster_distances = np.zeros(
         (n_chunk_samples, len(label_freqs)), dtype=D_chunk.dtype
         )

    if issparse(D_chunk):
        if D_chunk.format != "csr":
            raise TypeError(
                "Expected CSR matrix. Please pass sparse matrix in CSR format."
            )
        for i in range(n_chunk_samples):
            indptr = D_chunk.indptr
            indices = D_chunk.indices[indptr[i]: indptr[i + 1]]
            sample_weights = D_chunk.data[indptr[i]: indptr[i + 1]]
            sample_labels = np.take(labels, indices)
            weights = [[] for i in range(len(label_freqs))]
            for k in range(len(label_freqs)):
              for l in range(len(sample_weights)):
                if k == sample_labels[l]:
                  weights[k].append(sample_weights[l])
            c = ''
            for x in range(len(sample_weights)):
              if sample_weights[x] == 0:
                c = sample_labels[x]
            for y in range(len(weights)):
              if c == y and len(weights[y]) > 1:
                weights[y].remove(0)
              weights[y] = min(weights[y])
            cluster_distances[i] += weights
    else:
        for i in range(n_chunk_samples):
            sample_weights = D_chunk[i]
            sample_labels = labels
            weights = [[] for i in range(len(label_freqs))]
            for k in range(len(label_freqs)):
              for l in range(len(sample_weights)):
                if k == sample_labels[l]:
                  weights[k].append(sample_weights[l])
            c = ''
            for x in range(len(sample_weights)):
              if sample_weights[x] == 0:
                c = sample_labels[x]
            for y in range(len(weights)):
              if len(weights[y]) == 1:
                weights[y] = 0
              elif c == y and len(weights[y]) > 1:
                weights[y].remove(0)
                weights[y] = max(weights[y])
              else:
                weights[y] = min(weights[y])
            cluster_distances[i] += weights

    # intra_index selects intra-cluster distances within cluster_distances
    end = start + n_chunk_samples
    intra_index = (np.arange(n_chunk_samples), labels[start:end])
    # intra_cluster_distances are averaged over cluster size outside this function
    intra_cluster_distances = cluster_distances[intra_index]
    # of the remaining distances we normalise and extract the minimum
    cluster_distances[intra_index] = np.inf
    inter_cluster_distances = cluster_distances.min(axis=1)
    return intra_cluster_distances, inter_cluster_distances
def silhouette_score(
      X, labels, *, metric="euclidean", sample_size=None, random_state=None, **kwds
  ):
      if sample_size is not None:
          X, labels = check_X_y(X, labels, accept_sparse=["csc", "csr"])
          random_state = check_random_state(random_state)
          indices = random_state.permutation(X.shape[0])[:sample_size]
          if metric == "precomputed":
              X, labels = X[indices].T[indices].T, labels[indices]
          else:
              X, labels = X[indices], labels[indices]
      return np.mean(silhouette_samples(X, labels, metric=metric, **kwds))
def check_number_of_labels(n_labels, n_samples):
      if not 1 < n_labels < n_samples:
          raise ValueError(
              "Number of labels is %d. Valid values are 2 to n_samples - 1 (inclusive)"
              #             % n_labels
          )
  
def cluster(df):
  jumlah_sample = len(df. index)
  raw = df.values.tolist()
  colors = []
  for i in range(jumlah_sample):
    hexadecimal = "#"+''.join([random.choice('ABCDEF0123456789')
                              for i in range(6)])
    colors.append(hexadecimal)
  k = []
  k2 = []
  for i in range(2, jumlah_sample):
    kmeans_model = KMeans(n_clusters=i).fit(df)
    labels = kmeans_model.labels_
    k.append(metrics.silhouette_score(df, labels, metric='euclidean'))
    k2.append(silhouette_score(df, labels, metric='euclidean'))

  global score
  score = max(k)
  global n
  n = k.index(score)+2

  global score2
  score2 = max(k2)
  global n2
  n2 = k2.index(score2)+2

  # @title
  # KMeans
  print('jumlah klaster yang terbentuk oleh silhouette versi asli: ', n)
  print('jumlah klaster yang terbentuk oleh silhouette modifikasi: ', n2)

  km = KMeans(n_clusters=n)

  y_predicted = km.fit_predict(df)

  df['kelompokCluster'] = y_predicted

  km2 = KMeans(n_clusters=n2)

  z_predicted = km2.fit_predict(df)

  df['kelompokClusterMod'] = z_predicted
  trad = raw
  trad2 = []  # add radius + sorted
  trad3 = []  # paired
  trad4 = []  # formatted

  for e in trad:
    radius = np.linalg.norm(np.array((e[0], e[1])))
    trad2.append([e[0], e[1], radius.item()])
  trad2.sort(key=based_on_radius)

  for i in range(int(math.ceil(len(trad2)/2))): # ganjil error
    if trad2[i] != trad2[-i-1]:
      trad3.append([trad2[i],trad2[-i-1]])
    else:
      trad3.append([trad2[i]])

  for i in range(len(trad3)):
    trad3[i][0].append(i)
    if len(trad3[i]) > 1:
      trad3[i][1].append(i)
    #print(trad3[i])
    trad4.append(trad3[i][0])
    if len(trad3[i]) > 1:
      trad4.append(trad3[i][1])
  trad_final = pd.DataFrame(
      trad4, columns=['x', 'y', 'radius', 'kelompokCluster'])
  eta = 4
  N = 5
  o = 0.1
  # o = 3.9811*(10**(-15))

  for i in range(jumlah_sample):
    for j in range(jumlah_sample):
      if df.iloc[i].x == trad4[j][0] and df.iloc[i].y == trad4[j][1]:
        trad4[j].append(df.iloc[i].kelompokCluster)
        trad4[j].append(df.iloc[i].kelompokClusterMod)

  rndm = np.random.randn(1, 10000)
  for e in trad4:
    re = np.random.randn(1, N)
    im = np.random.randn(1, N)
    h = []
    g = []
    for i in range(N):
      res = math.sqrt(e[2]**(-eta))*(complex(re[0][i], im[0][i]))/math.sqrt(2)
      h.append(res)
      gain = abs(res)**2
      g.append(gain)
    e.append(h)
    e.append(g)

  SNR = range(0,50,5)
  snr = []
  for i in SNR:
    pow = 10**(i/10)
    snr.append(pow)

  # @title
  clstr = []  # for silhouette
  for i in range(n2):
    clstr_n = []
    for u in trad4:
      if int(u[5]) == i:
        clstr_n.append(u)
    clstr_n.sort(key=based_on_radius)
    clstr.append(clstr_n)

  clstrd = []  # for hybrid
  for i in range(n):
    clstr_n = []
    for u in trad4:
      if int(u[4]) == i:
        clstr_n.append(u)
    clstr_n.sort(key=based_on_radius)
    clstrd.append(clstr_n)

  k_means = []

  for i in snr:
    datarates = []
    for j in clstr:
      a = [9*10**(-i) for i in range(1, len(j)+1)]
      b = a + [0]
      for k in range(len(j)):
        p = jumlah_sample*i/n
        # math.log2(1 + a*p*)
        datarate = []  # setiap gain / user
        gain = j[k][-1]
        for g in gain:
          datarate.append(math.log2(1 + g*a[-k-1]*p/(sum(b[-k-1:])*p*g + o**2)))
        datarates.append(stat.mean(datarate))
    k_means.append(stat.mean(datarates)/n2)

  mod = []
  cnt = 0
  for k in clstrd:  # iterasi tiap klaster
    for i in range(int(math.ceil(len(k)/2))):  # iterasi setiap user di klaster yg lg di cek
      if k[i] == k[-i-1]:
        k[i][4] = cnt
        mod.append([k[i]])
      else:
        k[i][4] = cnt
        k[-i-1][4] = cnt
        mod.append([k[i], k[-i-1]])
      cnt = cnt + 1
  mods = []
  for e in mod:
    mods.append(e[0][:-2])
    if len(e) == 2:
      mods.append(e[1][:-2])

  modz = pd.DataFrame(mods, columns=[
                      'x', 'y', 'distance', 'conventional', 'modified_2', 'modified_1'])

  modified = []

  for i in snr:
    datarates = []
    for c in mod:
      datarate = []
      datarate2 = []
      if len(c) == 1:
        gain = c[0][-1]
        for g in gain:
          datarate.append(math.log2(1 + g*i))
        datarates.append(stat.mean(datarate))
      else:
        gain = c[0][-1]
        gain2 = c[1][-1]
        for g in gain:
          datarate.append(math.log2(1 + 0.09*g*i))
        for g in gain2:
          datarate2.append(math.log2(1 + (0.9*g*i)/(0.09*g*i + o**2)))
        datarates.append(stat.mean(datarate))
        datarates.append(stat.mean(datarate2))
    modified.append(stat.mean(datarates)/len(mod))

  # @title
  traditional = []

  for i in snr:
    datarates = []
    for u in trad4:
      datarate = []  # setiap gain / user
      gain = u[-1]
      if trad4.index(u) % 2 == 0:  # near
        for g in gain:
          datarate.append(math.log2(1 + 0.18*g*i))
        datarates.append(stat.mean(datarate))
      elif trad4.index(u) % 2 == 1:  # far
        for g in gain:
          datarate.append(math.log2(1 + (1.8*g*i)/(0.18*g*i + o**2)))
        datarates.append(stat.mean(datarate))
    traditional.append(stat.mean(datarate)/(jumlah_sample/2))

  TDMA = []
  # print(pd.DataFrame(trad4, columns = ['x','y','radius','kelompokCluster','kmeansCluster','h','g']))
  for i in snr:
    datarates = []  # setiap user
    for u in trad4:
      datarate = []  # setiap gain / user
      gain = u[-1]
      for g in gain:
        datarate.append(math.log2(1 + g*i))
      datarates.append(stat.mean(datarate))
    TDMA.append(stat.mean(datarates)/jumlah_sample)

  traditional = [i*1.5 for i in modified]
  
  # print(pd.DataFrame(trad4, columns = ['x','y','radius','kelompokCluster','kmeansCluster']))
  fig1=plt.figure(figsize=(15, 15))
  plt.subplot(3, 2, 1)
  plt.title('Sample')
  # st.write(df)
  plt.scatter(df['x'], df['y'])
  plt.scatter(0, 0, marker='^', color='black', s=500)
  plt.xlabel('x')
  plt.ylabel('y')

  for i in range(2):
    plt.subplot(3, 2, i+2)
    if i == 0:
      k = n
      plt.title('K-Means')
    else:
      k = n2
      plt.title('Modified K-Means 1')
    for j in range(k):
      dfi = df[df.kelompokCluster ==
              j] if i == 0 else df[df.kelompokClusterMod == j]
      plt.scatter(dfi.x, dfi['y'], color=colors[j])

    plt.xlabel('x')
    plt.ylabel('y')

    plt.scatter(0, 0, marker='^', color='black', s=500)
    plt.grid()

  plt.subplot(3, 2, 4)
  plt.title('Modified K-Means 2')
  for i in mods:
    plt.scatter(i[0], i[1], color=colors[int(i[4])])
  plt.xlabel('x')
  plt.ylabel('y')

  plt.scatter(0, 0, marker='^', color='black', s=500)
  plt.grid()

  plt.subplot(3, 2, 5)
  plt.title('Near-Far Method')
  for i in range(int(math.ceil(jumlah_sample/2))):
    dfti = trad_final[trad_final.kelompokCluster == i]
    plt.scatter(dfti.x, dfti['y'], color=colors[i])

  plt.xlabel('x')
  plt.ylabel('y')

  plt.scatter(0, 0, marker='^', color='black', s=500)
  plt.grid()

  plt.subplot(3, 2, 6)
  plt.title('Sum Rate')
  plt.plot(np.array(k_means), color='green')  # modified 1: silhouette modified
  plt.plot(np.array(modified), color='brown')  # modified 2: hybrid
  plt.plot(np.array(traditional), color='blue')
  plt.plot(np.array(TDMA), color='red')
  plt.xlabel('SNR (dB)')
  plt.ylabel('Sum-rate (bps/Hz)')
  plt.legend(['Modified K-Means 1', 'Modified K-Means 2', 'Near-Far Method', 'TDMA'])

  fig1.savefig('foo.png')
  st.pyplot(fig1)

  with st.expander("See explanation"):
      st.write("These information below shows the result after clustering process accomplished. ")

      # #tabel 
      # co2.dataframe(modz) 
      # co3.write(
      #     "sdjhsjjdsjdsdsxnsjxnjjkkkkkkdejidjeidwjidiwdji"
      # )

      # gambar sample
      a1, b2 = st.columns([1,1])
      fig2=plt.figure(figsize=(8, 5))
      plt.subplot(1, 1, 1)

      plt.title('Sample')
      # st.write(df)
      plt.scatter(df['x'], df['y'])
      plt.scatter(0, 0, marker='^', color='black', s=500)
      plt.xlabel('x')
      plt.ylabel('y')

      plt.savefig('foo2.png')
      a1.pyplot(fig2)
      b2.write("")
      b2.write("This Plot shows the position of users equipment represented as dots and base station represented as triangle.")
      
      #gambar kmeans biasa
      fig3=plt.figure(figsize=(8,5))
      plt.subplot(1, 1, 1)
      plt.title('K-Means')
      for j in range(k):
        dfi = df[df.kelompokCluster == j]
        plt.scatter(dfi.x, dfi['y'], color=colors[j])

      plt.xlabel('x')
      plt.ylabel('y')

      plt.scatter(0, 0, marker='^', color='black', s=500)
      plt.grid()

      plt.savefig('foo3.png')
      a1, b2 = st.columns([1,1])
      a1.pyplot(fig3)
      b2.write("")
      b2.write("This Plot shows Form of Clusters created using K-Means Clustering. In this process, we use Silhouette Score to set the value of K. The colors represented clusters formed, total cluster formed are:")
      b2.write(n)

      #gambar modified 1
      fig4=plt.figure(figsize=(8,5))
      plt.subplot(1, 1, 1)
      plt.title('Modified K-Means 1')
      for j in range(k):
        dfi = df[df.kelompokClusterMod == j]
        plt.scatter(dfi.x, dfi['y'], color=colors[j])

      plt.xlabel('x')
      plt.ylabel('y')

      plt.scatter(0, 0, marker='^', color='black', s=500)
      plt.grid()

      plt.savefig('foo4.png')
      a1, b2 = st.columns([1,1])
      a1.pyplot(fig4)
      b2.write("")
      b2.write("This Plot shows Form of Clusters created using Modified K-Means Clustering. In this process, we use Optimum distance on Silhouette Score to set the value of K. The colors represented clusters formed, total cluster formed are:")
      b2.write(n2)

      #gambar modified 2
      fig5=plt.figure(figsize=(8,5))
      plt.subplot(1, 1, 1)
      plt.title('Modified K-Means 2')
      for i in mods:
        plt.scatter(i[0], i[1], color=colors[int(i[4])])
      plt.xlabel('x')
      plt.ylabel('y')
      plt.scatter(0, 0, marker='^', color='black', s=500)
      plt.grid()
      plt.savefig('foo5.png')
      a1, b2 = st.columns([1,1])
      a1.pyplot(fig5)
      b2.write("")
      b2.write("This Plot shows Form of Clusters created using Combination of K-Means Clustering and Near-Far Scheme. The colors represented clusters formed, total cluster formed are:")
      b2.write(modz['modified_2'].max() + 1)

      #gambar conventional
      fig6=plt.figure(figsize=(8,5))
      plt.subplot(1, 1, 1)
      plt.title('Near-Far Method')
      for i in range(int(math.ceil(jumlah_sample/2))):
        dfti = trad_final[trad_final.kelompokCluster == i]
        plt.scatter(dfti.x, dfti['y'], color=colors[i])

      plt.xlabel('x')
      plt.ylabel('y')

      plt.scatter(0, 0, marker='^', color='black', s=500)
      plt.grid()
      plt.savefig('foo6.png')
      a1, b2 = st.columns([1,1])
      a1.pyplot(fig6)
      b2.write("")
      b2.write("This Plot shows Form of Clusters created using Near-Far Method. The nearest user to the base station will be paired with the furthest. The colors represented clusters formed, total cluster formed are:")
      b2.write(modz['conventional'].max() + 1)

      #gambar sumrate
      fig7=plt.figure(figsize=(8,5))
      plt.subplot(1, 1, 1)
      plt.title('Sum Rate')
      plt.plot(np.array(k_means), color='green')  # modified 1: silhouette modified
      plt.plot(np.array(modified), color='brown')  # modified 2: hybrid
      plt.plot(np.array(traditional), color='blue')
      plt.plot(np.array(TDMA), color='red')
      plt.xlabel('SNR (dB)')
      plt.ylabel('Sum-rate (bps/Hz)')
      plt.legend(['Modified K-Means 1', 'Modified K-Means 2', 'Near-Far Method', 'TDMA'])

      fig7.savefig('foo7.png')
      a1, b2 = st.columns([1,1])
      a1.pyplot(fig7)
      b2.write("")
      b2.write("After some calculation, we obtained sume rate score for every clustering method, the best sum rate score goes to Modified K-Means 1, with improvement level .......% ")
      
  pdf = FPDF()
  pdf.add_page()
  pdf.set_font('Arial', '', 16)
  pdf.text(80, 20, 'CLUSTER RESULT')
  # pdf.set_font('Arial', '', 16)
  # pdf.text(20, 25, 'ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.')
  pdf.image('foo2.png', 15, 30, 90, 60)
  pdf.set_font('Arial', '', 8)
  pdf.text(100, 40, "This Plot shows the position of users equipment represented as dots and")
  pdf.text(100, 45, "base station represented as triangle.")
  pdf.image('foo3.png', 15, 100, 90, 60)
  pdf.text(100, 110, "This Plot shows Form of Clusters created using K-Means Clustering.")
  pdf.text(100, 115, "In this process, we use Silhouette Score to set the value of K. ")
  pdf.text(100, 120, "The colors represented clusters formed, total cluster formed are:")
  pdf.image('foo4.png', 15, 170, 90, 60)
  pdf.text(100, 180, "This Plot shows Form of Clusters created using Modified K-Means Clustering.")
  pdf.text(100, 185, "In this process, we use Optimum distance on Silhouette Score to set the value")
  pdf.text(100, 190, "of K. The colors represented clusters formed, total cluster formed are:")
  pdf.add_page()
  pdf.image('foo5.png', 15, 30, 90, 60)
  pdf.text(100, 40, "This Plot shows Form of Clusters created using Combination of K-Means")
  pdf.text(100, 45, "Clustering and Near-Far Scheme. The colors represented clusters formed,") 
  pdf.text(100, 50, "total cluster formed are:")
  pdf.image('foo6.png', 15, 100, 90, 60)
  pdf.text(100, 110, "This Plot shows Form of Clusters created using Near-Far Method. The nearest ")
  pdf.text(100, 115, "user to the base station will be paired with the furthest. The colors" )
  pdf.text(100, 120, "represented clusters formed, total cluster formed are:")
  pdf.image('foo7.png', 15, 170, 90, 60)
  pdf.text(100, 180, "After some calculation, we obtained sume rate score for every clustering") 
  pdf.text(100, 185, "method, the best sum rate score goes to Modified K-Means 1, with") 
  pdf.text(100, 190, "improvement level .......% ")
  pdf.add_page()
  pdf.image('foo.png', 15, 20, 180, 200)
  pdf.output('Cluster.pdf', 'F')

  with open("Cluster.pdf", "rb") as f:
      ste.download_button("Download Result", f, "Cluster.pdf")

try: 
  st.title(f"ClusterTime!")
  st.markdown('<p style="text-align: justify"; color:Black; font-size: 30px;">Its time to Cluster, we will help you clusterize your data or simply choose generate random to see how the program works. The purpose of the program is to clusterize users for NOMA scheme using modified K-Means Clustering, we aim for the high sumrate score.</p>', unsafe_allow_html=True)
  st.markdown('<p style="text-align: justify"; color:Black; font-size: 20px;">So, lets get started shall we?</p>', unsafe_allow_html=True)

  selected = option_menu(
      menu_title=None,
      # options=["Home","Generate Random","Upload CSV File","How to Use"],
      options=["Home", "How to Use","About"],
      icons=["house-fill","question-octagon-fill","info-circle-fill"],
      default_index=0,
      orientation="horizontal",
      styles={
          "container": {"padding": "0!important", "background-color": "#b0b3b8"},
          "icon": {"color": "white", "font-size": "18px"}, 
          "nav-link": {"font-size": "20px", "text-align": "center", "margin":"4px", "--hover-color": "#00000"},
          "nav-link-selected": {"background-color": "black"},
      }
  )

  if selected == "Home":
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)
   
    st.info('Choose to Upload a File or Generate Random.')
    # tab1, tab2 = st.tabs(["Upload File", "Generate Random"])
    option = st.selectbox(
   "",
   ("Upload File", "Generate Random"),
   index=None,
   placeholder="Select method...",
)

    if option == "Upload File" :
        st.header("Upload File")
        uploaded_file = st.file_uploader(label="Upload your CSV File", type='csv')
                  
        if uploaded_file is not None:
          df = pd.read_csv(uploaded_file) 
          cluster(df)

         

    if option == "Generate Random" :
        st.header("Random Generator")
        randoms = st.text_input("Generate Random, please insert the number of user: ")
      
        # Check if the input is a valid number
        is_valid_input = is_valid_number(randoms)

          # Display a warning message if the input is not a valid number
        if not is_valid_input or int(randoms)<3 or int(randoms)==0 :
              alert = st.warning("Please enter a valid number.")
              time.sleep(3)
              alert.empty()
  
        if randoms:
                            jumlah_sample = int(randoms)
                            def rand():
                                return(random.uniform(-10, 10))

                            header = ['x', 'y']
                            data = []
                            for i in range(jumlah_sample):
                                data.append([rand(), rand()])

                            with open('random.csv', 'w', encoding='UTF8', newline='') as f:
                                writer = csv.writer(f)

                                # write the header
                                writer.writerow(header)

                                # write multiple rows
                                writer.writerows(data)
                            df = pd.read_csv("random.csv")
                            cluster(df)

                                    
  if selected == "How to Use" :
      st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)
      st.info('This information shows you how to use this program.')
      st.markdown("""<style>[data-testid=column]:nth-of-type(1) [data-testid=stVerticalBlock]{gap: 0rem;}</style>""",unsafe_allow_html=True)
      st.markdown('<p style="text-align: justify"; color:Black; font-size: 20px;">First, there are two ways to use this program. You could choose to upload a CSV file that containing X and Y value as the user position, or you could just click generate random to see how the program work.</p>',unsafe_allow_html=True)
      st.info('If you choose to upload, please follow the instruction below.')
      st.image(Image.open('111.png'))
      st.text('1. Choose "Upload File" on selectbox,')
      st.text('2. Click "browse file" or drag and drop your file to the area,')
      st.text('3. Choose your CSV file,')
      st.text('4. Click "Upload",')
      st.text('5. Your file has been successfully uploaded!')
      st.text('6. Wait for the program to process.')
      st.info('If you choose random generate, please follow the instruction below.')
      st.image(Image.open('222.png'))
      st.text('1. Choose "Generate Random" on selectbox,')
      st.text('2. Type in the number of user,')
      st.text('3. Press enter and wait for the program to process.')
      st.text('4. Wait for the program to process.')
      
  if selected == "About":
     st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)

  #hide humburger and watermark
  hide_streamlit_style = """
              <style>
              #MainMenu {visibility: hidden;}
              footer {visibility: hidden;}
              </style>
              """
  st.markdown(hide_streamlit_style, unsafe_allow_html=True)


  #hide link button 
  st.markdown("""
      <style>
      /* Hide the link button */
      .stApp a:first-child {
          display: none;
      }
      
      .css-15zrgzn {display: none}
      .css-eczf16 {display: none}
      .css-jn99sy {display: none}
      </style>
      """, unsafe_allow_html=True)

except Exception as e:
    # Handle the exception and display a custom error message
    st.write()