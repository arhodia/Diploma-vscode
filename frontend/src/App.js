import FacultyList from "./FacultyList";

function App() {
  return (
    <div className="w-full flex flex-col items-center mt-8">
      <h1 className="text-3xl font-bold mb-8">File Upload Demo</h1>

      <div className="w-full max-w-4xl">
        <FacultyList />
      </div>
    </div>
  );
}

export default App;
