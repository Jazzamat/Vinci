
import React from 'react'
import {Grid} from '@mui/material/'

import './App.css';





function App() {
  return (
    <div className='container'>

      <nav className='navbar'>
          <li><a href="#home">Home</a></li> 
          <li><a href="#about">about</a></li>
          <li><a href="#create">Create</a></li> 
      </nav>
      <section id="home">
          <header className="wrapper"> </header>
          <div className="content">
              <h1 className='title-text' >Vinci</h1>
          </div>
      </section>

   

      <section id="about">
        <h1 className='section h1'>About Vinci</h1>
          <Grid 
          container  
          spacing={0}
          direction="column"
          alignItems="center"
          justifyContent="center"
        >
            
          <Grid item xs={12} sm={6} md={3}>
            <div className='rainbow-image-container'> 
              <img className='display-image' width="500" height="500"  style={{opacity:0}}/>
            </div>
          </Grid>

          {/* <div className='image-container'> 
            <img className='display-image' width="500" height="500"/>
          </div> */}
          </Grid>
        <p className='section p'>The vision for vinci is similar to dall-e by Open-Ai, but instead of a text to image converter its a song to image generator. The plan is to have a deep neural network learn the pattern between song and cover art, learning from extensive list of songs and their corresponding art online, from sites like spotify, apple music and so on, generating stunning artwork to match the character and feel of the song </p>
  
      </section>

      <section id="create">
          <h1 className='title-text'>Create</h1>
      </section>


    </div>
  );
}

export default App;