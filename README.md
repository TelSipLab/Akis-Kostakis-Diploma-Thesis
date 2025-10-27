# Akis-Kostakis-Diploma-Thesis
Diploma Thesis Files for Akis Kostakis

# Run

Make sure eigen eigen is available to known compiler paths (/usr/local/include, /usr/include).  
Compiles and runs the app.  

```bash
make clean
make all -j

# Choose what to run
./bin/mahony.out
```


# clangd

Many times VSCode has very bad intelisense so switch to clangd.  
Install the extension (clangd), together with the cpp extension.  
Modify the settings  
```json
{
    // Disable Microsoft C++ IntelliSense
    "C_Cpp.intelliSenseEngine": "disabled",
    "C_Cpp.autocomplete": "disabled",
    "C_Cpp.errorSquiggles": "disabled",
    
    // clangd settings
    "clangd.arguments": [
        "--background-index",
        "--clang-tidy",
        "--header-insertion=iwyu",
        "--completion-style=detailed",
        "--function-arg-placeholders",
        "--fallback-style=llvm"
    ],
    "clangd.path": "/usr/bin/clangd",
    
    // Keep C++ debugger functionality
    "C_Cpp.debugShortcut": true
}
```


Generate the compile_commands.json  

```bash
sudo apt install bear
make clean # Important we need to rebuild otherwise bear wont capture the build
bear -- make all
cat compile_commands.json
```
