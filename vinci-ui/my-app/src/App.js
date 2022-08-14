import logo from './logo.svg';
import './App.css';
import { SearchBarInput } from './components/SearchBar';
import {SearchBarButton} from './components/SearchBar';


function App() {
  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <p>
          Welcome to Vinci. The devs are at it...
        </p>

        
      </header>
      <SearchBarInput/>
    </div>
  );
}

export default App;
