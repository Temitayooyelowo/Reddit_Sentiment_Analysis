const Search = ({ searchQuery, setSearchQuery }) => (
    <form action="/" method="get">
      <input
        value={searchQuery}
        onInput={(e) => setSearchQuery(e.target.value)}
        type="text"
        id="header-search"
        placeholder="Search stocks"
        name="s"
      />
      <button type="submit">Search</button>
    </form>
  );
  
  export default Search;