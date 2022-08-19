import logo from './logo.svg';
import './App.css';


import React from 'react';

function App() {
  return (
    <div className='container'>

      <nav className='navbar'>
          <li><a href="#home">Home</a></li> 
          <li><a href="#create">Create</a></li> 
          <li><a href="#about">about</a></li> 
      </nav>


      <section id="home">
          <header className="wrapper"> </header>
          <div className="content">
              <h1 className='title-text' >Vinci</h1>
          </div>
      </section>

      <section id="create">
        <h1 className='title-text'>Create</h1>
      </section>

      <section id="about">
        <h1 className='title-text'>About</h1>
      </section>

    </div>
  );
}

export default App;