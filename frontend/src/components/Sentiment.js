const Sentiment = (sentiment) => {
  var emotion = "";

  if (sentiment.sentiment < 0) {
    emotion = " 🙁 ";
  } 
  
  if (sentiment.sentiment === 0) {
    emotion = " 😐 ";
  } 
  
  if (sentiment.sentiment > 0){
    emotion = " 😀 ";
  }

  return emotion;
};

export default Sentiment;