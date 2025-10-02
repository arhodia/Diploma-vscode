import { useEffect, useState } from "react";

export default function FacultyList() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/indian_faculty")
      .then(res => res.json())
      .then(setData)
      .catch(console.error);
  }, []);

  return (
    <div>
      <h1>Faculty Dataset</h1>
      <ul>
        {data.map((row, idx) => (
          <li key={idx}>{row['Vidwan-ID']} — {row['Department']}</li>
        ))}
      </ul>
    </div>
  );
}
