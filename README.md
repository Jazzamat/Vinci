![Alt text](https://github.com/Jazzamat/Vinci/blob/main/misc/VinciLogo.png)

[![Python 3.10.6](https://img.shields.io/badge/python-3.10.6-blue)](https://www.python.org/downloads/release/python-3106/)
[![Node.js 8.5.1](https://img.shields.io/badge/Node.js-8.5.1-green)](https://nodejs.org/docs/latest-v8.x/api/documentation.html)
[![](https://dcbadge.vercel.app/api/server/3GUjmazQNu)](https://discord.gg/3GUjmazQNu&style=flat)
# Vinci: The AI-Powered Album Cover Artist

Vinci is an ambitious project drawing inspiration from OpenAI's DALL-E. Unlike DALL-E which converts text to images, Vinci aims to transform songs into visual representations. It utilizes a deep learning network to uncover and understand the underlying connections between songs and their corresponding album art. With a comprehensive dataset of songs and related art sourced from online platforms such as Spotify and Apple Music, Vinci aspires to generate compelling album covers based on musical inputs.

## Getting Started

This project is built using React. To get started, follow the guide provided by [Create React App](https://create-react-app.dev/docs/getting-started/).

## Current Version: 0.01 Alpha

## Introducing AIG-Net-I: Audio-Image Generation Network I

The core of Vinci is its unique architecture, AIG-Net-I. This model endeavors to learn the intricate relationship between audio and image data. It achieves this by training on pairs of songs and their corresponding album cover art. 

Post-training, the system is capable of generating album art from new audio inputs. This is done by processing the audio through the Audio Encoder, passing it to the conditional model (GAN/Transformer), and finally through the Image Decoder. This pipeline helps Vinci in creating visually enticing and musically coherent album covers. 

Please refer to the `vinci-core/neural_networks/tensorflow/AIG-Net-I` directory for a deeper understanding of the AIG-Net-I architecture.

---

Keep in mind that this is an early-stage project under active development. All feedback and contributions are welcome. Join us as we endeavor to bridge the gap between music and visual art with the power of AI.
