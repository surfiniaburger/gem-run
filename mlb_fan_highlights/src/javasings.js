fetch('https://mlb-strings-1011675918473.us-central1.run.app/api/v1/podcast', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      team: "Dodgers",
      players: ["Shohei Ohtani", "Mookie Betts", "Freddie Freeman"],
      timeframe: "Last game",
      game_type: "Regular Season",
      language: "english",
      opponent: "Giants"
    }),
  })
  .then(response => response.json())
  .then(data => console.log(data))
  .catch((error) => console.error('Error:', error));