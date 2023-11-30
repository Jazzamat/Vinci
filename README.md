![Alt text](https://github.com/Jazzamat/Vinci/blob/main/misc/VinciLogo.png)

[![Python 3.10.6](https://img.shields.io/badge/python-3.10.6-blue)](https://www.python.org/downloads/release/python-3106/)
[![Node.js 8.5.1](https://img.shields.io/badge/Node.js-8.5.1-green)](https://nodejs.org/docs/latest-v8.x/api/documentation.html)
[![](https://dcbadge.vercel.app/api/server/3GUjmazQNu?style=flat)](https://discord.gg/3GUjmazQNu)
# Vinci: The ML Album Cover Artist

Vinci is an ambitious project drawing inspiration from OpenAI's DALL-E. Unlike DALL-E which converts text to images, Vinci aims to transform songs into visual representations. It utilizes a deep learning network to uncover and understand the underlying connections between songs and their corresponding album art. With a comprehensive dataset of songs and related art sourced from online platforms such as Spotify and Apple Music, Vinci aspires to generate compelling album covers based on musical inputs.

## Getting Started

This project is built using React. To get started, follow the guide provided by [Create React App](https://create-react-app.dev/docs/getting-started/).

## Current Version: 0.01 Alpha

## Introducing AIG-Net-I: Audio-Image Generation Network I

The core of Vinci is its unique architecture, AIG-Net-I. This model endeavors to learn the intricate relationship between audio and image data. It achieves this by training on pairs of songs and their corresponding album cover art. 

Post-training, the system is capable of generating album art from new audio inputs. This is done by processing the audio through the Audio Encoder, passing it to the conditional model (GAN/Transformer), and finally through the Image Decoder. This pipeline helps Vinci in creating visually enticing and musically coherent album covers.

The proposed architecture attempts to learn the underlying relationship between audio and image data by training on pairs of song audio and corresponding album art. After training, the system should be able to generate album art from new song audio by passing the audio through the Audio Encoder and then the conditional model (GAN/Transformer) and Image Decoder.

### Overview
AIG-Net-I is a cutting-edge deep learning model designed to capture and visually represent the relationship between audio and image data. At its core, this network transforms song audio into corresponding album art, creating a visual representation of music through a unique blend of auditory and visual processing.

### Architecture
The architecture of AIG-Net-I is composed of two main encoders – one for audio and the other for images. The Audio Encoder processes audio inputs, like spectrograms, transforming them into a latent space where key features are distilled into a compact form. Similarly, the Image Encoder translates visual elements from album art into this shared latent space, focusing on capturing the essence of visual data.

The heart of AIG-Net-I lies in its conditional model. This model acts as a bridge in the latent space, finding and learning the complex relationships between audio and visual features. It's here that the audio's latent representation is molded into a form that can be translated back into the visual domain.

The final step in the network is the Image Decoder. This component takes the audio-influenced latent representation and reconstructs it back into an image – effectively generating album art from the audio data. The transition from audio to image is a journey from one sensory world to another, guided by the learned correlations in the model.

Please refer to the `vinci-core/neural_networks/tensorflow/AIG-Net-I` directory for a deeper understanding of the AIG-Net-I architecture.

---

Keep in mind that this is an early-stage project under active development. All feedback and contributions are welcome. Join us as we endeavor to bridge the gap between music and visual art with the power of AI.
