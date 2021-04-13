import React, { useState } from "react";
import { Button, Dropdown, Form, Input } from "semantic-ui-react";

// const options = [
//   { value: 'past year', text: 'Past year' },
//   { value: 'past 3 months', text: 'Past 3 months'},
//   { value: 'past month', text: "Past month"},
//   { value: 'past week', text: "Past week"},
// ];

// const defaultOption = options[2];

// const TimeDropdown = () => (
// <Dropdown
//   placeholder='Select time'
//   selection
//   options={options}
// />
// )

const Search = () => {
  const [stock, setStock] = useState('');
  // const [time, setTime] = useState('');
  const time = 'past month'

  return (
    <Form>
      <Form.Field>
        <Input 
        placeholder="Search stocks"
        value={stock}
        onChange={e => setStock(e.target.value)}
        name="s"
        />
      </Form.Field>
      <Form.Field>
        <Button onClick={async () => {
            const query = {time, stock};
            const response = await fetch('http://127.0.0.1:5000/sentiment', {
              method: 'GET',
              body: JSON.stringify(query)
            })

            if (response.ok) {
              console.log("response worked")
            }
        }}>Submit</Button>
      </Form.Field>
    </Form>
  )
}

export default Search;