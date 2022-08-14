import React, {useState} from 'react';
import {Grid, TextField, Button} from '@mui/material';
import FilterAltIcon from '@mui/icons-material/FilterAlt';

/**
 * @name SearchBarInput
 * @description This component is the input field for any search bar rendered in the platform
 * @param {size field integer: x/12, function to define search term, search function handler}
 * @returns A SearchBarInput Component
 */
export const SearchBarInput = ({xs, defineSearchTerm, onSearch}) => {
    return (
        <div>
            blah
        </div>
    
        // <Grid item xs={xs} sx={{ px: 1 }}>
        //     <TextField 
        //       fullWidth 
        //       required
        //       label="where should I park my car..." 
        //       variant="outlined" 
        //       onChange={(event) => defineSearchTerm(event.target.value)}
        //       onKeyPress={(event) => {
        //         if (event.key === 'Enter'){
        //             onSearch();
        //         }
        //       }}
        //       />   
        // </Grid>
    );
}

/**
 * @name SearchBarButton
 * @description This component renders the search button next to the input field throughout the platform
 * @param {search function handler} 
 * @returns A SearchBarButton Component
 */
export const SearchBarButton = ({onSearch}) => {
    return (
        <Grid item alignItems="stretch" style={{ display: "flex" }} sx={{ px: 1 }}>
            <Button variant="contained" onClick={onSearch}>
                Search
            </Button>
        </Grid>
    );
}

export const SearchBarFilter = ({context, handleSearch}) => {

    const [filterDisplay, setFilterDisplay] = useState(false);

    return (
        <>
            <Grid item alignItems="center" style={{ display: "flex" }}>
                <FilterAltIcon 
                    fontSize="large" 
                    sx={{color: '#2196f3'}}
                    onClick={() => {setFilterDisplay(true)}}
                />
            </Grid>
        </>

    );
}