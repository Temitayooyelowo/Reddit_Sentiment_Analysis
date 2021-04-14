import { Header } from 'semantic-ui-react';

const Sentiment = (data) => {
  var emotion = "";

  if (data.sentiment === 'negative') {
    emotion = `${data.stock} is doing not so good 🙁 `;
  } else if (data.sentiment === 'neutral') {
    emotion = `${data.stock} is going so-so 😐`;
  } else if (data.sentiment === 'positive'){
    emotion = `${data.stock} is doing good! 😀 `;
  } else {
    emotion = "An error has occured"
  }

  return (
    <Header as='h2'>
      {emotion}
    </Header>);
};

export default Sentiment;