import React, { useState } from "react";
import Sentiment from './Sentiment';
import { Button, Container, Form, Input } from "semantic-ui-react";  
  
const Search = () => {
  const [stock, setStock] = useState('');
  const [time, setTime] = useState('past month');
  const [sentiment, setSentiment] = useState('');
  
  const TimeDropdown = () => ( 
    <select value={time} onChange={e => setTime(e.target.value)}>
      <option value="past week">Past week</option>
      <option selected value="past month">Past month</option>
      <option value="past 3 months">Past 3 months</option>
      <option value="past year">Past year</option>
    </select>
  )

  return (
    <>
      <Form>
        <Form.Group>
          <Form.Field>
            <Input 
            placeholder="Search stocks"
            value={stock}
            onChange={e => {setStock(e.target.value); setSentiment('')}}
            name="s"
            />
          </Form.Field>
        </Form.Group>
        <Form.Group>
          <Form.Field>
            <TimeDropdown/>
            <Button onClick={async () => {
              try {
                const response = await fetch(`http://127.0.0.1:5000/sentiment?time=${time}&ticker=${stock}`, {
                  method: 'GET'
                })

                const data = await response.json()

                if (response) {
                  setSentiment(data.sentiment)
                  console.log(data.sentiment)
                }
              } catch (e) {
                console.log('Error')
              }
            }}>Search</Button>
          </Form.Field>
        </Form.Group>
      </Form>
      {sentiment && <Container >
        <Sentiment 
          sentiment={sentiment}
          stock={stock}/>
      </Container> }
      </>
  )
}

export default Search;