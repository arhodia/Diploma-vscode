import { useEffect, useState } from "react";

export default function FacultyList() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch("http://127.0.0.1:5000/api/indian_faculty")
      .then(async (res) => {
        console.log("status", res.status);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        console.log("json length", json.length);
        console.log("sample row", json[0]);
        setData(json);
      })
      .catch((err) => {
        console.error("fetch error:", err);
      });
  }, []);

  return (
    <div>
      <h1>Faculty Dataset</h1>
      {!data.length ? <p>no data yet</p> : (
        <ul>
          {data.map((row, idx) => (
            <li key={idx}>
              {row.Name ?? "(no Name)"} — {row.Department ?? "(no Department)"}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
