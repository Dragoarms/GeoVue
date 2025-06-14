let
    // Get configuration from Excel table
    Config = Excel.CurrentWorkbook(){[Name="JSONConfig"]}[Content],
    
    // Function to get config value
    GetConfig = (item as text) => Config{[ConfigItem=item]}[FilePath],
    
    // Get file paths
    ReviewsPath = GetConfig("CompartmentReviews"),
    RegisterPath = GetConfig("CompartmentRegister"),
    OriginalPath = GetConfig("OriginalImages"),
    
    // Function to get field names from a JSON file with better error handling
    GetFieldsFromFile = (filePath as text, fileName as text) as record =>
        let
            // Check if it's a SharePoint URL
            IsSharePoint = Text.StartsWith(filePath, "http"),
            
            // Try to get content - this will trigger auth if needed
            FileContent = if IsSharePoint then
                    try 
                        Web.Contents(filePath, [
                            Headers = [Accept="application/json"],
                            ManualStatusHandling = {404, 403}
                        ]) 
                    otherwise null
                else
                    try File.Contents(filePath) otherwise null,
            
            // Check if we got content
            HasContent = FileContent <> null,
            
            // Parse JSON if we have content
            Json = if HasContent then 
                    try Json.Document(FileContent) otherwise null 
                else 
                    null,
            
            // Check if JSON parsed successfully
            JsonParsed = Json <> null,
            
            // Check if it's a list (expected format)
            IsList = Json is list,
            RecordCount = if IsList then List.Count(Json) else 0,
            
            // Extract fields
            Fields = if IsList and RecordCount > 0 then
                let
                    // Get first record to check structure
                    FirstRecord = Json{0},
                    IsRecord = FirstRecord is record,
                    
                    // Extract all unique field names
                    AllFields = List.Distinct(
                        List.Combine(
                            List.Transform(Json, each 
                                if _ is record then
                                    try Record.FieldNames(_) otherwise {}
                                else {}
                            )
                        )
                    )
                in
                    AllFields
            else {},
            
            // Return detailed info for debugging
            Result = [
                FileName = fileName,
                Path = filePath,
                IsSharePoint = IsSharePoint,
                HasContent = HasContent,
                JsonParsed = JsonParsed,
                IsList = IsList,
                RecordCount = RecordCount,
                FieldCount = List.Count(Fields),
                Fields = Fields,
                Error = if not HasContent then "Failed to read file"
                       else if not JsonParsed then "Failed to parse JSON"
                       else if not IsList then "JSON is not a list"
                       else if RecordCount = 0 then "No records in JSON"
                       else "OK"
            ]
        in
            Result,
    
    // Get fields from each file with debug info
    ReviewInfo = GetFieldsFromFile(ReviewsPath, "compartment_reviews.json"),
    RegisterInfo = GetFieldsFromFile(RegisterPath, "compartment_register.json"),
    OriginalInfo = GetFieldsFromFile(OriginalPath, "original_images_register.json"),
    
    // Extract just the fields
    ReviewFields = ReviewInfo[Fields],
    RegisterFields = RegisterInfo[Fields],
    OriginalFields = OriginalInfo[Fields],
    
    // Create debug table
    DebugTable = Table.FromRecords({ReviewInfo, RegisterInfo, OriginalInfo}),
    
    // Combine all unique fields
    AllFields = List.Sort(
        List.Distinct(ReviewFields & RegisterFields & OriginalFields)
    ),
    
    // Check if we have any fields
    HasFields = List.Count(AllFields) > 0,
    
    // Create main table or debug table
    OutputTable = if HasFields then
        let
            // Create field table
            FieldTable = Table.FromList(AllFields, Splitter.SplitByNothing(), {"FieldName"}),
            
            // Add columns showing which file contains each field
            WithReviews = Table.AddColumn(FieldTable, "compartment_reviews", 
                each List.Contains(ReviewFields, [FieldName])),
            
            WithRegister = Table.AddColumn(WithReviews, "compartment_register", 
                each List.Contains(RegisterFields, [FieldName])),
            
            WithOriginal = Table.AddColumn(WithRegister, "original_images_register", 
                each List.Contains(OriginalFields, [FieldName])),
            
            // Add data type column
            WithType = Table.AddColumn(WithOriginal, "DataType", each
                if Text.Contains([FieldName], "Date") then "DateTime"
                else if [FieldName] = "From" or [FieldName] = "To" or Text.Contains([FieldName], "_From") or Text.Contains([FieldName], "_To") then "Number"
                else if Text.StartsWith([FieldName], "Contains_") or Text.StartsWith([FieldName], "plus_") or Text.EndsWith([FieldName], "_Status") or Text.EndsWith([FieldName], "_Image") then "Boolean"
                else "Text"
            )
        in
            WithType
    else
        DebugTable  // Show debug info if no fields found
in
    OutputTable