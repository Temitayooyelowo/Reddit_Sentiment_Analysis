import React, { useEffect, useState } from "react";
import Search from "./components/Search";
import {Container, Header } from "semantic-ui-react";
import "./App.css";

function App() {
  // const { search } = window.location;
  // const query = new URLSearchParams(search).get("s");
  // const [searchQuery, setSearchQuery] = useState(query || "");
  // const [stocks, setStocks] = useState([]);
  // const [sentiment, setSentiment] = useState([]);

  // useEffect(() => {
  //   fetch("http://127.0.0.1:5000/sentiment").then(response =>
  //   response.json().then(data => {
  //     setStocks(data.sentiment)
  //   })
  //   );
  // }, []);

  return (
    <>
      <div>
        <Header as='h1'>Reddit Sentiment Analysis</Header>
      </div>
      <div style={{display: 'flex',  justifyContent:'center', alignItems:'center', height: '25vh'}}>
        <Container style={{alignItems: 'center'}}>
          <Search/>
        </Container>
      </div>
    </>
  )
}

export default App;
