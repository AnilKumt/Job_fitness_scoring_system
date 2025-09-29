import React, { useState, useEffect } from 'react';
import axios from 'axios';

export default function UsernameCheck() {
  const [username, setUsername] = useState('');
  const [response, setResponse] = useState('');
  const [debouncedUsername, setDebouncedUsername] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedUsername(username), 500);
    return () => clearTimeout(timer);
  }, [username]);

  useEffect(() => {
    if (!debouncedUsername) return;
    const fetchData = async () => {
      try {
        const res = await axios.get(`http://localhost:5000/user/${debouncedUsername}`);
        setResponse(res.data);
      } catch (err) {
        console.error(err);
        setResponse('Error fetching data');
      }
    };
    fetchData();
  }, [debouncedUsername]);

  return (
    <div className='w-1/2 h-1/2 shadow-2xl p-10'>
      <label className='text-2xl' htmlFor="username">
        Enter your username:
      </label>
      <input
        className='text-2xl border-2 ml-2 p-1'
        type="text"
        id='username'
        value={username}
        onChange={(e) => setUsername(e.target.value)}
      />

      {username && (
        <div className='mt-4 text-black text-3xl'>
          <p>
            Backend response is : <span><b><i>{response}</i></b></span>
          </p>
        </div>
      )}
    </div>
  );
}
