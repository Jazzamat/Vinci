
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
              <p >alpha v0.1</p>
          </div>
      </section>

   

      <section id="about">
       
            <h1 className='section h1'>About Vinci</h1>
        
            <div className='image-container-container'>
                <div className='rainbow-image-container'> 
                  <img className='display-image' width="430" height="422"  style={{opacity:0}}/>
                </div>

                <div className='image-container'> 
                  <img className='display-image' src="https://i1.sndcdn.com/artworks-KWGxuS0opzs0yb7B-iV37Tg-t500x500.jpg" width="430" height="422"  style={{opacity:1}}/>
                  <img className='display-image-2' src="https://www.metallica.com/dw/image/v2/BCPJ_PRD/on/demandware.static/-/Sites-Metallica-Library/default/dwed9e0db5/images/releases/20150807_213911_7549_752890.jpg?sw=372&sh=372&sm=cut&sfrm=jpeg&q=95" width="430" height="422"  style={{opacity:1}}/>
                  {/* <img className='display-image-3' src="https://lh3.googleusercontent.com/dcxXIIlest09vnvKznWM9VWQXu1EL7lKxBzXGzwgmVjmMNBm1dEWT_0qn1xrEZYyKF_qRE1TLq8P_JY_mQ=w544-h544-l90-rj" width="430" height="422"  style={{opacity:1}}/> */}

                </div>
           
           
             
                
            </div>
            <p className='section p'>The vision for vinci is similar to dall-e by Open-Ai, but instead of a text to image converter its a song to image generator. The plan is to have a deep neural network learn the pattern between song and cover art, learning from extensive list of songs and their corresponding art online, from sites like spotify, apple music and so on, generating stunning artwork to match the character and feel of the song </p>

       
      </section>

      <section id="create">
          <h1 className='title-text'>Create</h1>
      </section>


    </div>
  );
}

export default App;