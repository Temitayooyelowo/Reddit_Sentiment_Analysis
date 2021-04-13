import React, { useState } from "react";
import Search from "./components/Search";
import Sentiment from "./components/Sentiment";
import "./App.css";

const stocks = [
  { id: "1", name: "Amazon", sentiment: -1 },
  { id: "2", name: "Shopify", sentiment: +1 },
  { id: "3", name: "Google", sentiment: 0 },
  { id: "4", name: "Facebook", sentiment: 1 },
];

const filterStocks = (stocks, query) => {
  if (!query) {
    return stocks;
  }

  return stocks.filter((post) => {
    const postName = post.name.toLowerCase();
    return postName.includes(query);
  });
};

// const options = [
//   { value: 'past year', label: 'Past year' },
//   { value: 'past 3 months', label: 'Past 3 months'},
//   { value: 'past month', label: "Past month"},
//   { value: 'past week', label: "Past week"},
// ];

// const defaultOption = options[2];

const App = () => {
  const { search } = window.location;
  const query = new URLSearchParams(search).get("s");
  const [searchQuery, setSearchQuery] = useState(query || "");
  const filteredStocks = filterStocks(stocks, searchQuery);

  return (
    <div>
      <Search searchQuery={searchQuery} setSearchQuery={setSearchQuery} />
      {/* <Dropdown options={options} value={defaultOption} placeholder="Select an option" />; */}
      <ul>
        {filteredStocks.map((stock) => (
          <p key={stock.key}>
            {stock.name}
            <Sentiment sentiment={stock.sentiment} />
          </p>
        ))}
      </ul>
    </div>
  );
};

export default App;
