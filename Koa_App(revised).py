{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/maverickxlr/Koa_App/blob/main/Koa_App(revised).py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "id": "fPgYPCUzVxWE"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "from keras.models import Model\n",
        "from google.colab import drive\n",
        "from keras.utils.image_utils import img_to_array\n",
        "from skimage import io\n",
        "from numpy import asarray\n",
        "from PIL import Image"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "6aFWsub9Oyyh",
        "outputId": "c2f19270-6968-4730-b34a-da76da3ddc4c"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/\n",
            "Collecting streamlit\n",
            "  Downloading streamlit-1.22.0-py2.py3-none-any.whl (8.9 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.9/8.9 MB\u001b[0m \u001b[31m76.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: altair<5,>=3.2.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.2.2)\n",
            "Collecting blinker>=1.0.0 (from streamlit)\n",
            "  Downloading blinker-1.6.2-py3-none-any.whl (13 kB)\n",
            "Requirement already satisfied: cachetools>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (5.3.0)\n",
            "Requirement already satisfied: click>=7.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.1.3)\n",
            "Requirement already satisfied: importlib-metadata>=1.4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.6.0)\n",
            "Requirement already satisfied: numpy in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.22.4)\n",
            "Requirement already satisfied: packaging>=14.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (23.1)\n",
            "Requirement already satisfied: pandas<3,>=0.25 in /usr/local/lib/python3.10/dist-packages (from streamlit) (1.5.3)\n",
            "Requirement already satisfied: pillow>=6.2.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.4.0)\n",
            "Requirement already satisfied: protobuf<4,>=3.12 in /usr/local/lib/python3.10/dist-packages (from streamlit) (3.20.3)\n",
            "Requirement already satisfied: pyarrow>=4.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (9.0.0)\n",
            "Collecting pympler>=0.9 (from streamlit)\n",
            "  Downloading Pympler-1.0.1-py3-none-any.whl (164 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m164.8/164.8 kB\u001b[0m \u001b[31m17.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: python-dateutil in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.8.2)\n",
            "Requirement already satisfied: requests>=2.4 in /usr/local/lib/python3.10/dist-packages (from streamlit) (2.27.1)\n",
            "Requirement already satisfied: rich>=10.11.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (13.3.4)\n",
            "Requirement already satisfied: tenacity<9,>=8.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (8.2.2)\n",
            "Requirement already satisfied: toml in /usr/local/lib/python3.10/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions>=3.10.0.0 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.5.0)\n",
            "Requirement already satisfied: tzlocal>=1.1 in /usr/local/lib/python3.10/dist-packages (from streamlit) (4.3)\n",
            "Collecting validators>=0.2 (from streamlit)\n",
            "  Downloading validators-0.20.0.tar.gz (30 kB)\n",
            "  Preparing metadata (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "Collecting gitpython!=3.1.19 (from streamlit)\n",
            "  Downloading GitPython-3.1.31-py3-none-any.whl (184 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m184.3/184.3 kB\u001b[0m \u001b[31m15.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting pydeck>=0.1.dev5 (from streamlit)\n",
            "  Downloading pydeck-0.8.1b0-py2.py3-none-any.whl (4.8 MB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m82.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: tornado>=6.0.3 in /usr/local/lib/python3.10/dist-packages (from streamlit) (6.2)\n",
            "Collecting watchdog (from streamlit)\n",
            "  Downloading watchdog-3.0.0-py3-none-manylinux2014_x86_64.whl (82 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m82.1/82.1 kB\u001b[0m \u001b[31m10.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: entrypoints in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit) (0.4)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit) (3.1.2)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit) (4.3.3)\n",
            "Requirement already satisfied: toolz in /usr/local/lib/python3.10/dist-packages (from altair<5,>=3.2.0->streamlit) (0.12.0)\n",
            "Collecting gitdb<5,>=4.0.1 (from gitpython!=3.1.19->streamlit)\n",
            "  Downloading gitdb-4.0.10-py3-none-any.whl (62 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m6.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.10/dist-packages (from importlib-metadata>=1.4->streamlit) (3.15.0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas<3,>=0.25->streamlit) (2022.7.1)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil->streamlit) (1.16.0)\n",
            "Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit) (1.26.15)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit) (2022.12.7)\n",
            "Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit) (2.0.12)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.10/dist-packages (from requests>=2.4->streamlit) (3.4)\n",
            "Requirement already satisfied: markdown-it-py<3.0.0,>=2.2.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->streamlit) (2.2.0)\n",
            "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /usr/local/lib/python3.10/dist-packages (from rich>=10.11.0->streamlit) (2.14.0)\n",
            "Requirement already satisfied: pytz-deprecation-shim in /usr/local/lib/python3.10/dist-packages (from tzlocal>=1.1->streamlit) (0.1.0.post0)\n",
            "Requirement already satisfied: decorator>=3.4.0 in /usr/local/lib/python3.10/dist-packages (from validators>=0.2->streamlit) (4.4.2)\n",
            "Collecting smmap<6,>=3.0.1 (from gitdb<5,>=4.0.1->gitpython!=3.1.19->streamlit)\n",
            "  Downloading smmap-5.0.0-py3-none-any.whl (24 kB)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->altair<5,>=3.2.0->streamlit) (2.1.2)\n",
            "Requirement already satisfied: attrs>=17.4.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (23.1.0)\n",
            "Requirement already satisfied: pyrsistent!=0.17.0,!=0.17.1,!=0.17.2,>=0.14.0 in /usr/local/lib/python3.10/dist-packages (from jsonschema>=3.0->altair<5,>=3.2.0->streamlit) (0.19.3)\n",
            "Requirement already satisfied: mdurl~=0.1 in /usr/local/lib/python3.10/dist-packages (from markdown-it-py<3.0.0,>=2.2.0->rich>=10.11.0->streamlit) (0.1.2)\n",
            "Requirement already satisfied: tzdata in /usr/local/lib/python3.10/dist-packages (from pytz-deprecation-shim->tzlocal>=1.1->streamlit) (2023.3)\n",
            "Building wheels for collected packages: validators\n",
            "  Building wheel for validators (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "  Created wheel for validators: filename=validators-0.20.0-py3-none-any.whl size=19579 sha256=479e4c06df8c5ca6bbbb60a96494a6d89139b47e9bf31c3b06e58b8f2e9e7473\n",
            "  Stored in directory: /root/.cache/pip/wheels/f2/ed/dd/d3a556ad245ef9dc570c6bcd2f22886d17b0b408dd3bbb9ac3\n",
            "Successfully built validators\n",
            "Installing collected packages: watchdog, validators, smmap, pympler, blinker, pydeck, gitdb, gitpython, streamlit\n",
            "Successfully installed blinker-1.6.2 gitdb-4.0.10 gitpython-3.1.31 pydeck-0.8.1b0 pympler-1.0.1 smmap-5.0.0 streamlit-1.22.0 validators-0.20.0 watchdog-3.0.0\n"
          ]
        }
      ],
      "source": [
        "!pip install streamlit \n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Um7OSY_mkVgq"
      },
      "outputs": [],
      "source": [
        "#!pip install --upgrade streamlit"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jfLmUkY_ygdL"
      },
      "outputs": [],
      "source": [
        "#!pip install pyngrok"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "o7EC3o5iO1lV",
        "outputId": "403a2884-26a2-4717-e01c-bfa1e5cdb3fd"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oXgxtFXzpvZo",
        "outputId": "b07ed1e1-38d4-4d0c-dcd5-6984d38d685f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Writing app1.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app1.py \n",
        "import streamlit as st\n",
        "import cv2\n",
        "import io\n",
        "from PIL import Image\n",
        "import tensorflow as tf\n",
        "from tensorflow import keras\n",
        "import numpy as np\n",
        "from keras.utils.image_utils import img_to_array\n",
        "\n",
        "\n",
        "\n",
        "map_dict1 = {0: 'Healthy knee image'}\n",
        "map_dict2 = {0: 'Grade 1 (Doubtful)',\n",
        "             1: 'Grade 2 (Minimal)',\n",
        "             2: 'Grade 3 (Moderate)',\n",
        "             3: 'Grade 4 (Severe)'}\n",
        "\n",
        "\n",
        "\n",
        "def image_preprocessing(uploaded_file):\n",
        "    image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)\n",
        "    image = cv2.imdecode(image,1)\n",
        "    image = Image.fromarray(image)        \n",
        "    image = image.resize((224,224)) \n",
        "    image= img_to_array(image)\n",
        "    image= np.expand_dims(image, axis=0)\n",
        "    return image\n",
        "\n",
        "@st.cache_resource\n",
        "def load_binarymodel():\n",
        "    binary_model = tf.keras.models.load_model('/content/drive/MyDrive/Saved_models/Resnet_binary/weights-improvement-36-0.94.hdf5')\n",
        "    return binary_model\n",
        "\n",
        "@st.cache_resource\n",
        "def load_multiclassmodel():\n",
        "    multiclass_model = tf.keras.models.load_model('/content/drive/MyDrive/Saved_models/Resnet_multiclass_onehotencoded/weights-improvement-26-1.00.hdf5.hdf5')\n",
        "    return multiclass_model\n",
        "\n",
        "\n",
        "#st.title(\"Knee Osteoarthritis Detector\")\n",
        "st.image(\"/content/drive/MyDrive/Screenshot logo 2.png\")\n",
        "st.text(\"This application predicts Knee Osteoarthritis severity by analyzing X-Ray images.\")\n",
        "st.sidebar.title(\"About This App\")\n",
        "st.sidebar.markdown(\"The application determines the severity of knee osteoarthritis on the basis of KL grade classification, by using machine learning algorithms.\")\n",
        "st.sidebar.title(\"Classification Grades\")\n",
        "st.sidebar.header(\"Grade 0 (none):\")\n",
        "st.sidebar.markdown(\"Definite absence of x-ray changes of osteoarthritis\")\n",
        "st.sidebar.header(\"Grade 1 (doubtful):\")\n",
        "st.sidebar.markdown(\"Doubtful joint space narrowing and possible osteophytic lipping\")\n",
        "st.sidebar.header(\"Grade 2 (minimal):\")\n",
        "st.sidebar.markdown(\"Definite osteophytes and possible joint space narrowing\")\n",
        "st.sidebar.header(\"Grade 3 (moderate):\")\n",
        "st.sidebar.markdown(\"Moderate multiple osteophytes, definite narrowing of joint space and some sclerosis and possible deformity of bone ends\")\n",
        "st.sidebar.header(\"Grade 4 (severe):\")\n",
        "st.sidebar.markdown(\"Large osteophytes, marked narrowing of joint space, severe sclerosis and definite deformity of bone ends\")\n",
        "#st.sidebar.image(\"/content/drive/MyDrive/The-KL-grading-system-to-assess-the-severity-of-knee-OA-Source.png\",width=None,use_column_width='never')\n",
        "\n",
        "### load file\n",
        "uploaded_file = st.file_uploader(\"Please upload an X-Ray image\", type=['png','jpeg','jpg'])\n",
        " \n",
        "#picture = st.camera_input(\"Take a picture\")\n",
        "\n",
        "if uploaded_file is not None:\n",
        "    st.image(uploaded_file, caption='Uploaded Image.')\n",
        "\n",
        "#if picture is not None:\n",
        "    #st.image(picture,caption='Captured Image')    \n",
        "\n",
        "Generate_pred = st.button(\"Generate Prediction\")   \n",
        "\n",
        "if uploaded_file is not None:\n",
        "   if Generate_pred:\n",
        "      image=image_preprocessing(uploaded_file)\n",
        "\n",
        "      binary_prediction = load_binarymodel().predict(image)\n",
        "      binary_pred = (binary_prediction>=0.5).astype(int)\n",
        "  \n",
        "      if (binary_pred<0.5):\n",
        "          st.title(\"Healthy knee image\")\n",
        "      else:\n",
        "          multiclass_prediction = load_multiclassmodel().predict(image).argmax()\n",
        "          st.title(\"Prediction: {}\".format(map_dict2 [multiclass_prediction])) \n",
        "\n",
        "#if picture is not None:\n",
        "   #if Generate_pred:\n",
        "      #image=image_preprocessing(picture)\n",
        "\n",
        "      #binary_prediction = load_binarymodel().predict(image)\n",
        "      #binary_pred = (binary_prediction>=0.5).astype(int)\n",
        "  \n",
        "      #if (binary_pred<0.5):\n",
        "          #st.title(\"Healthy knee image\")\n",
        "      #else:\n",
        "          #multiclass_prediction = load_multiclassmodel().predict(image).argmax()\n",
        "          #st.title(\"Prediction: {}\".format(map_dict2 [multiclass_prediction])) \n",
        "          \n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ulGrT0fd-jJJ"
      },
      "outputs": [],
      "source": [
        "#from pyngrok import ngrok\n",
        "#ngrok.set_auth_token(\"2OeFs65fVdnXWfT7l2qU6ZxZDvy_5tsPNwaGxvNMDHJkvRnk1\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "tz9CsBRWHoE4",
        "outputId": "1c996227-feba-4eff-fc6b-7fe875c13b54"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[##................] | fetchMetadata: sill resolveWithNewModule color-convert@2\u001b[0m\u001b[K\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.\n",
            "\u001b[0m\n",
            "\u001b[K\u001b[?25hnpx: installed 22 in 5.455s\n",
            "your url is: https://strong-bugs-travel-34-147-42-218.loca.lt\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.147.42.218:8501\u001b[0m\n",
            "\u001b[0m\n",
            "2023-05-01 18:22:53.569245: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n",
            "1/1 [==============================] - 1s 1s/step\n",
            "1/1 [==============================] - 1s 926ms/step\n",
            "1/1 [==============================] - 1s 1s/step\n",
            "1/1 [==============================] - 1s 563ms/step\n",
            "1/1 [==============================] - 1s 533ms/step\n"
          ]
        }
      ],
      "source": [
        "!streamlit run app1.py & npx localtunnel --port 8501\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "khufW65oAY1h"
      },
      "outputs": [],
      "source": [
        "#!nohup streamlit run app.py --server.port 80  & \n",
        "#url = ngrok.connect()\n",
        "#print(url) #generates our URL\n",
        "#streamlit run --server.port 9095 app.py >/dev/null #used for starting our server"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1vAotaQhZTydxMitfSGTrMQKGyvuqcMoc",
      "authorship_tag": "ABX9TyOiaP+iooWGF45w7EWm1Gp0",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}