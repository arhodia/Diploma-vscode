import React, { useState } from "react";
import { RadioGroup, FormControlLabel, Radio } from '@mui/material';
import { 
  Select, MenuItem, InputLabel, FormControl, Button, Box, Typography, Alert, CircularProgress, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper
} from "@mui/material";

const STATIC_OPTIONS = [
 'business products   services',
 'consumer products   services',
 'it management',
 'real estate',
 'financial services',
 'engineering',
 'security',
 'logistics   transportation', 
 'insurance',
 'telecommunications',
 'manufacturing', 
 'travel   hospitality', 
 'software', 
 'construction',
 'environmental services', 
 'health', 
 'education', 
 'advertising   marketing', 
 'human resources', 
 'food   beverage', 
 'government services',
 'media', 
 'energy',
 'retail',
 'it system development', 
 'it services', 
 'computer hardware'
];

export default function FacultyList() {
  const [selectedOption, setSelectedOption] = useState('');
  const [fileType, setFileType] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadMessage, setUploadMessage] = useState({ type: '', text: '' });
  const [results, setResults] = useState([]);
  const [recommendations, setRecommendations] = useState([]);
  const [algorithm, setAlgorithm] = useState("");

  const handleAlgorithmChange = (e) => {
  setAlgorithm(e.target.value);
  };

  // Dropdown option change
  const handleOptionChange = (e) => {
    setSelectedOption(e.target.value);
  };


  // Radio button choice for file type
  const handleFileTypeChange = (e) => {   
    setFileType(e.target.value);
  };

  // Upload (POST)
  const handleUpload = async () => {
    if (!selectedOption || !fileType || !algorithm) {
      setUploadMessage({type: 'error', text: 'Select a file AND a workspace!'});
      return;
    }

    setLoading(true);
    setUploadMessage({ type: '', text: '' });

    const formData = new FormData();
    formData.append('selected_option', selectedOption);
    formData.append('file_type', fileType);
    formData.append("algorithm", algorithm); 

    try {
      const response = await fetch('http://127.0.0.1:5000/api/upload_files', {
        method: 'POST',
        body: formData
      });

      const result = await response.json();

      if (response.ok) {
        setUploadMessage({ type: 'success', text: result.message || 'File uploaded successfully!' });
        setResults(result.results || []);
        setRecommendations(result.recommendations || []);// μόνο μετά το POST
      } else {
        setUploadMessage({ type: 'error', text: result.error || 'Error during upload.' });
      }
    } catch (error) {
      setUploadMessage({ type: 'error', text: 'Connection error with the server' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: "80%", margin: "0 auto", paddingTop: "2em" }}>
      <h2 style={{ textAlign: "center" }}>Faculty Dataset</h2>
      {/* Dropdown */}
      <FormControl fullWidth margin="normal">
        <InputLabel id="dropdown-label">Select Option</InputLabel>
        <Select
          labelId="dropdown-label"
          value={selectedOption}
          label="Select Option"
          onChange={handleOptionChange}
        >
          {STATIC_OPTIONS.map(option => (
            <MenuItem key={option} value={option}>{option}</MenuItem>
          ))}
        </Select>
      </FormControl>

      {/* choose algorithm Section */}
      <Box
  sx={{
    marginTop: 3,
    marginBottom: 3,
    padding: 3,
    border: "1px solid #ddd",
    borderRadius: 2,
    backgroundColor: "#f9f9f9",
  }}
>
  {/* 3 sections σε μία γραμμή */}
  <Box
    sx={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
      gap: 2,
      flexWrap: "wrap", // για μικρές οθόνες να “σπάει” ωραία
    }}
  >
    {/* Αριστερά: Radio */}
    <FormControl component="fieldset">
      <RadioGroup
        row
        value={fileType || ""}
        onChange={handleFileTypeChange}
        name="file_type"
        sx={{ gap: 2 }}
      >
        <FormControlLabel value="start-up" control={<Radio />} label="Start-up" />
        <FormControlLabel value="researcher" control={<Radio />} label="Researcher" />
      </RadioGroup>
    </FormControl>

    {/* Μέση: Select αλγορίθμου */}
    <FormControl size="small" sx={{ minWidth: 220 }}>
      <InputLabel id="algorithm-label">Algorithm</InputLabel>
      <Select
        labelId="algorithm-label"
        value={algorithm}
        label="Algorithm"
        onChange={handleAlgorithmChange}
      >
        <MenuItem value="LSH">LSH</MenuItem>
        <MenuItem value="KMEANS">K-MEANS</MenuItem>
        <MenuItem value="BISECTING_KMEANS">K-MEANS BISECTING</MenuItem>
        <MenuItem value="DBSCAN">DBSCAN</MenuItem>
      </Select>
    </FormControl>

    {/* Δεξιά: Button */}
    <Button
      variant="contained"
      color="success"
      onClick={handleUpload}
      disabled={loading || !selectedOption || !fileType || !algorithm}
      sx={{ height: 40 }} 
    >
      {loading ? <CircularProgress size={24} /> : "Search"}
    </Button>
  </Box>

  {/* Upload Message */}
  {uploadMessage.text && (
    <Alert
      severity={uploadMessage.type}
      sx={{ marginTop: 2 }}
      onClose={() => setUploadMessage({ type: "", text: "" })}
    >
      {uploadMessage.text}
    </Alert>
  )}
</Box>



      {/* Data Table */}
      {results.length > 0 && (
  <TableContainer component={Paper} style={{ marginTop: "2em" }}>
    <Typography variant="h6" sx={{ padding: 2, fontWeight: "bold" }}>
      Top Results
    </Typography>

    <Table>
      <TableHead>
        <TableRow>
          <TableCell><strong>Name - Surname / Company</strong></TableCell>
          <TableCell><strong>Profile</strong></TableCell>
          <TableCell><strong>Research Field</strong></TableCell>
          <TableCell><strong>Field</strong></TableCell>
        </TableRow>
      </TableHead>

      <TableBody>
        {results.map((row, idx) => (
          <TableRow key={idx}>
            <TableCell>{row.name} - {row.surname} / {row.company_name}</TableCell>
            <TableCell>{row.profile}</TableCell>
            <TableCell>{row.topic_merged_norm}</TableCell>
            <TableCell>{row.cluster}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
)}


{/* Προτεινόμενα Αποτελέσματα */}
{recommendations.length > 0 && (
  <TableContainer component={Paper} style={{ marginTop: "2em" }}>
    <Typography variant="h6" sx={{ padding: 2, fontWeight: "bold" }}>
      Recommended Results
    </Typography>

    <Table>
      <TableHead>
        <TableRow>
          <TableCell><strong>Name - Surname / Company</strong></TableCell>
          <TableCell><strong>Profile</strong></TableCell>
          <TableCell><strong>Research Field</strong></TableCell>
          <TableCell><strong>Field</strong></TableCell>
        </TableRow>
      </TableHead>

      <TableBody>
        {recommendations.map((row, idx) => (
          <TableRow key={idx}>
            <TableCell>{row.name} - {row.surname} / {row.company_name}</TableCell>
            <TableCell>{row.profile}</TableCell>
            <TableCell>{row.topic_merged_norm}</TableCell>
            <TableCell>{row.cluster}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  </TableContainer>
)}
    </div>
  );
}