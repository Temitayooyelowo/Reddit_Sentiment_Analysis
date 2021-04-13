import React, { useEffect, useState } from "react";
import Search from "./components/Search";
import {Container } from "semantic-ui-react";
import "./App.css";

// const stocks = [
//   { id: "1", name: "Amazon", sentiment: -1 },
//   { id: "2", name: "Shopify", sentiment: +1 },
//   { id: "3", name: "Google", sentiment: 0 },
//   { id: "4", name: "Facebook", sentiment: 1 },
// ];

// const filterStocks = (stocks, query) => {
//   if (!query) {
//     return stocks;
//   }

//   return stocks.filter((post) => {
//     const postName = post.name.toLowerCase();
//     return postName.includes(query);
//   });
// };


// const App = () => {
//   const { search } = window.location;
//   const query = new URLSearchParams(search).get("s");
//   const [searchQuery, setSearchQuery] = useState(query || "");
//   const filteredStocks = filterStocks(stocks, searchQuery);

//   return (
//     <div>
//       <Search searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
//       {/* <Dropdown options={options} value={defaultOption} placeholder="Select an option" />; */}
//       <ul>
//         {filteredStocks.map((stock) => (
//           <p key={stock.key}>
//             {stock.name}
//             <Sentiment sentiment={stock.sentiment} />
//           </p>
//         ))}
//       </ul>
//     </div>
//   );
// };

function App() {
  // const { search } = window.location;
  // const query = new URLSearchParams(search).get("s");
  // const [searchQuery, setSearchQuery] = useState(query || "");
  const [stocks, setStocks] = useState([]);
  const [sentiment, setSentiment] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/sentiment").then(response =>
    response.json().then(data => {
      setStocks(data.sentiment)
    })
    );
  }, []);

  return (
    <Container style={{alignItems: 'center'}}>
      <Search/>
    </Container>
  )
}

export default App;
