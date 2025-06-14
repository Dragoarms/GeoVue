// Function: LoadJsonFromConfig
// Retrieves a JSON file path from the JSONConfig table by key and loads it

let
    LoadJsonFromConfig = (configKey as text) as table =>
    let
        // Load config table from named range
        configTable = Excel.CurrentWorkbook(){[Name="JSONConfig"]}[Content],
        
        // Filter to find the matching config item
        row = Table.SelectRows(configTable, each [ConfigItem] = configKey),
        
        // Error if not found
        _ = if Table.IsEmpty(row) then error "Config key not found: " & configKey else null,

        // Extract the path string
        jsonPath = row{0}[FilePath],

        // Load and parse the JSON file
        json = Json.Document(File.Contents(jsonPath)),

        // Return as a table
        result = Table.FromList(json, Splitter.SplitByNothing(), null, null, ExtraValues.Error)
    in
        result
in
    LoadJsonFromConfig