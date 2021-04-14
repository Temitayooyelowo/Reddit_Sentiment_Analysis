import React from "react";
import Search from "./components/Search";
import { Container, Header, Image} from "semantic-ui-react";
import "./App.css";

function App() {
  return (
    <>
      <div>
        <Image src="https://logodownload.org/wp-content/uploads/2018/02/reddit-logo-16.png" width="100"/>
        <Header as='h1'> Stock Sentiment Analysis</Header>
      </div>
      <div style={{display: 'flex',  justifyContent:'center', alignItems:'top', height:'60vh', paddingTop:'50px'}}>
        <Container style={{alignItems: 'center'}}>
          <Search/>
        </Container>
      </div>
    </>
  )
}

export default App;
