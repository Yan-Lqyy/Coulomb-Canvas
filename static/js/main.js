// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const distributionsInput = document.getElementById('distributionsInput');

    const pointXInput = document.getElementById('pointX');
    const pointYInput = document.getElementById('pointY');
    const pointZInput = document.getElementById('pointZ');
    const calculatePointBtn = document.getElementById('calculatePointBtn');
    const pointResultDiv = document.getElementById('pointResult');

    const sliceAxisSelect = document.getElementById('sliceAxis');
    const sliceValueInput = document.getElementById('sliceValue');
    const sliceDim1RangeInput = document.getElementById('sliceDim1Range');
    const sliceDim2RangeInput = document.getElementById('sliceDim2Range');
    const sliceResolutionInput = document.getElementById('sliceResolution');
    const calculateSliceBtn = document.getElementById('calculateSliceBtn');

    const volumeXRangeInput = document.getElementById('volumeXRange');
    const volumeYRangeInput = document.getElementById('volumeYRange');
    const volumeZRangeInput = document.getElementById('volumeZRange');
    const volumeResolutionInput = document.getElementById('volumeResolution');
    const calculateVolumeBtn = document.getElementById('calculateVolumeBtn');

    const plotDiv2D = document.getElementById('plotDiv2D');
    const plotDiv3D = document.getElementById('plotDiv3D');
    const statusMessage = document.getElementById('statusMessage');

    // Hide plots initially
    plotDiv2D.style.display = 'none';
    plotDiv3D.style.display = 'none';

    function displayError(message) {
        statusMessage.textContent = `Error: ${message}`;
        statusMessage.style.color = 'red';
        console.error("User Input Error:", message);
    }

    function displayStatus(message) {
        statusMessage.textContent = message;
        statusMessage.style.color = '#6c757d'; // Default status color
    }

    function getDistributions() {
        try {
            const distributions = JSON.parse(distributionsInput.value);
            if (!Array.isArray(distributions)) {
                displayError('Distributions must be a valid JSON array.');
                return null;
            }
            if (distributions.some(d => typeof d !== 'object' || d === null || !('type' in d))) {
                displayError('Each distribution in the JSON array must be an object with a "type" property.');
                return null;
            }
            return distributions;
        } catch (e) {
            displayError(`Invalid JSON for distributions: ${e.message}`);
            return null;
        }
    }

    function parseRange(rangeStr, fieldName = "Range") {
        try {
            const parts = rangeStr.split(',').map(s => parseFloat(s.trim()));
            if (parts.length !== 2 || parts.some(isNaN)) {
                throw new Error("Invalid format. Expected two numbers separated by a comma (e.g., -5, 5).");
            }
            if (parts[0] > parts[1]) {
                return [parts[1], parts[0]];
            }
            return parts;
        } catch (e) {
            displayError(`Invalid ${fieldName}: ${e.message}`);
            return null;
        }
    }

    function getNumericInput(inputElement, fieldName, minValue = -Infinity, maxValue = Infinity, allowFloat = true) {
        const valueStr = inputElement.value;
        const value = allowFloat ? parseFloat(valueStr) : parseInt(valueStr, 10);

        if (isNaN(value)) {
            displayError(`${fieldName} must be a valid number.`);
            return null;
        }
        if (value < minValue) {
            displayError(`${fieldName} must be at least ${minValue}.`);
            return null;
        }
        if (value > maxValue) {
            displayError(`${fieldName} must be at most ${maxValue}.`);
            return null;
        }
        return value;
    }


    async function fetchData(endpoint, payload) {
        displayStatus('Calculating...');
        pointResultDiv.innerHTML = '';
        plotDiv2D.style.display = 'none';
        plotDiv3D.style.display = 'none';
        Plotly.purge(plotDiv2D);
        Plotly.purge(plotDiv3D);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', },
                body: JSON.stringify(payload),
            });
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: `Server error: ${response.status}` }));
                throw new Error(errorData.error || `Server error: ${response.status}`);
            }
            displayStatus('Calculation complete.');
            return await response.json();
        } catch (error) {
            console.error('API Fetch error:', error);
            displayError(`API Error: ${error.message}`);
            return null;
        }
    }

    calculatePointBtn.addEventListener('click', async () => {
        const distributions = getDistributions();
        if (!distributions) return;

        const pointX = getNumericInput(pointXInput, "Point X coordinate");
        const pointY = getNumericInput(pointYInput, "Point Y coordinate");
        const pointZ = getNumericInput(pointZInput, "Point Z coordinate");

        if (pointX === null || pointY === null || pointZ === null) return;

        const payload = {
            distributions: distributions,
            point: { x: pointX, y: pointY, z: pointZ }
        };

        const result = await fetchData('/api/calculate_field_at_point', payload);
        if (result) {
            pointResultDiv.innerHTML = `
                <h5>Field at Point (${pointX.toFixed(2)}, ${pointY.toFixed(2)}, ${pointZ.toFixed(2)}):</h5>
                <ul>
                    <li><strong>Ex:</strong> ${result.Ex.toExponential(3)} N/C</li>
                    <li><strong>Ey:</strong> ${result.Ey.toExponential(3)} N/C</li>
                    <li><strong>Ez:</strong> ${result.Ez.toExponential(3)} N/C</li>
                    <li><strong>Magnitude |E|:</strong> ${result.E_magnitude.toExponential(3)} N/C</li>
                    <li><strong>Potential V:</strong> ${result.potential.toExponential(3)} V</li>
                </ul>
            `;
            plotDiv2D.style.display = 'none';
            plotDiv3D.style.display = 'none';
        }
    });

    calculateSliceBtn.addEventListener('click', async () => {
        const distributions = getDistributions();
        if (!distributions) return;

        const sliceAxis = sliceAxisSelect.value;
        const sliceValue = getNumericInput(sliceValueInput, "Slice Value");
        const dim1Range = parseRange(sliceDim1RangeInput.value, "Slice Dimension 1 Range");
        const dim2Range = parseRange(sliceDim2RangeInput.value, "Slice Dimension 2 Range");
        const resolution = getNumericInput(sliceResolutionInput, "Slice Resolution", 2, 200, false);

        if (sliceValue === null || dim1Range === null || dim2Range === null || resolution === null) return;

        let sliceDefinition = {
            axis: sliceAxis, value: sliceValue, resolution: resolution
        };

        if (sliceAxis === 'x') {
            sliceDefinition.y_range = dim1Range; sliceDefinition.z_range = dim2Range;
        } else if (sliceAxis === 'y') {
            sliceDefinition.x_range = dim1Range; sliceDefinition.z_range = dim2Range;
        } else {
            sliceDefinition.x_range = dim1Range; sliceDefinition.y_range = dim2Range;
        }

        const payload = { distributions: distributions, slice_definition: sliceDefinition };
        const result = await fetchData('/api/calculate_field_on_slice', payload);

        if (result) {
            pointResultDiv.innerHTML = '';
            Plotly.purge(plotDiv3D);
            plotDiv3D.style.display = 'none';

            const points_dim1 = result.points_dim1;
            const points_dim2 = result.points_dim2;
            const vectors_dim1 = result.vectors_dim1;
            const vectors_dim2 = result.vectors_dim2;

            const range1_min = Math.min(...points_dim1); const range1_max = Math.max(...points_dim1);
            const range2_min = Math.min(...points_dim2); const range2_max = Math.max(...points_dim2);
            const range1_span = range1_max - range1_min;
            const range2_span = range2_max - range2_min;

            const effective_res_dim1 = (new Set(points_dim1)).size;
            const effective_res_dim2 = (new Set(points_dim2)).size;

            const spacing1 = range1_span / (effective_res_dim1 > 1 ? effective_res_dim1 - 1 : 1);
            const spacing2 = range2_span / (effective_res_dim2 > 1 ? effective_res_dim2 - 1 : 1);
            const estimated_spacing = Math.max(spacing1, spacing2) || 1;

            const vectorScaleFactor = 0.25;
            let quiverLinesX = [];
            let quiverLinesY = [];
            const magnitudes = vectors_dim1.map((u, i) => Math.sqrt(u*u + vectors_dim2[i]*vectors_dim2[i]));
            const max_magnitude = Math.max(0, ...magnitudes);

            for(let i=0; i < points_dim1.length; i++) {
                const magnitude = magnitudes[i];
                let scaled_u = 0, scaled_v = 0;
                if (magnitude > 1e-12) {
                    scaled_u = vectors_dim1[i] / magnitude * estimated_spacing * vectorScaleFactor;
                    scaled_v = vectors_dim2[i] / magnitude * estimated_spacing * vectorScaleFactor;
                }
                quiverLinesX.push(points_dim1[i], points_dim1[i] + scaled_u, null);
                quiverLinesY.push(points_dim2[i], points_dim2[i] + scaled_v, null);
            }

            const vectorLinesTrace = {
                 x: quiverLinesX, y: quiverLinesY, mode: 'lines', type: 'scatter',
                 line: { color: 'rgba(200,0,0,0.8)', width: 2 },
                 showlegend: false, hoverinfo: 'none'
            };

            const markerTrace = {
                x: points_dim1, y: points_dim2, mode: 'markers', type: 'scatter',
                marker: {
                    size: 6, color: magnitudes, colorscale: 'Plasma',
                    cmin: 0, cmax: max_magnitude > 0 ? max_magnitude : 1e-9,
                    colorbar: { title: 'Field Mag.' },
                    line: { color: 'rgba(0,0,0,0.5)', width: 0.5 }
                },
                 name: `Magnitude |E|`,
                 hovertemplate: result.axis_labels[0].toUpperCase() + ': %{x:.2f}<br>' +
                                result.axis_labels[1].toUpperCase() + ': %{y:.2f}<br>' +
                                '|E|: %{marker.color:.3e}<br>' +
                                'E' + result.axis_labels[0] + ': %{customdata[0]:.3e}<br>' +
                                'E' + result.axis_labels[1] + ': %{customdata[1]:.3e}<extra></extra>',
                customdata: vectors_dim1.map((v1, i) => [v1, vectors_dim2[i]])
            };

            const layout = {
                title: `Electric Field on ${result.slice_axis.toUpperCase()}=${result.slice_value.toFixed(2)} Slice`,
                xaxis: { title: result.axis_labels[0].toUpperCase() + ' (m)', scaleanchor: "y", scaleratio: 1 },
                yaxis: { title: result.axis_labels[1].toUpperCase() + ' (m)' },
                width: 600, height: 500, hovermode: 'closest',
                margin: {t: 50, b: 50, l: 50, r: 20}, autosize: true,
            };
            Plotly.newPlot(plotDiv2D, [vectorLinesTrace, markerTrace], layout, { responsive: true });
            plotDiv2D.style.display = 'block';
        }
    });

    calculateVolumeBtn.addEventListener('click', async () => {
        const distributions = getDistributions();
        if (!distributions) return;

        const xRange = parseRange(volumeXRangeInput.value, "Volume X Range");
        const yRange = parseRange(volumeYRangeInput.value, "Volume Y Range");
        const zRange = parseRange(volumeZRangeInput.value, "Volume Z Range");
        const resolution = getNumericInput(volumeResolutionInput, "Volume Resolution", 2, 25, false); // Reduced max res further for larger default cones

        if (xRange === null || yRange === null || zRange === null || resolution === null) return;

        const payload = {
            distributions: distributions,
            volume_definition: { x_range: xRange, y_range: yRange, z_range: zRange, resolution: resolution }
        };

        const result = await fetchData('/api/calculate_field_in_volume', payload);
        if (result) {
            pointResultDiv.innerHTML = '';
            Plotly.purge(plotDiv2D);
            plotDiv2D.style.display = 'none';

            const u_orig = result.vectors_u;
            const v_orig = result.vectors_v;
            const w_orig = result.vectors_w;
            const magnitudes = u_orig.map((u, i) => Math.sqrt(u*u + v_orig[i]*v_orig[i] + w_orig[i]*w_orig[i]));
            const max_magnitude = Math.max(0, ...magnitudes);

            // --- Cone Sizing Logic: `sizeref` determines the length of the LARGEST cone ---
            const rangeX_span = xRange[1]-xRange[0];
            const rangeY_span = yRange[1]-yRange[0];
            const rangeZ_span = zRange[1]-zRange[0];
            const minPlotDimension = Math.min(rangeX_span, rangeY_span, rangeZ_span);
            const maxPlotDimension = Math.max(rangeX_span, rangeY_span, rangeZ_span);

            let coneSizerefValue;
            // Target: Make the largest cone have a length that's a noticeable fraction of the plot dimension.
            // This fraction should be large enough to make the vectors conspicuous.
            // Let's aim for the largest cone to be, for example, 1/2 or 1/3 of the smallest plot dimension.
            // Or, make it proportional to the average spacing between grid points.
            const avgGridSpacing = ( (rangeX_span/(resolution-1||1)) + (rangeY_span/(resolution-1||1)) + (rangeZ_span/(resolution-1||1)) ) / 3;
            
            if (avgGridSpacing > 1e-9) {
                // Let the largest cone be roughly 1 to 1.5 times the average grid spacing.
                // This ensures cones are prominent relative to their density.
                coneSizerefValue = avgGridSpacing * 1.5; // ADJUST THIS FACTOR (e.g., 1.0, 1.5, 2.0)
            } else if (minPlotDimension > 1e-9) {
                // Fallback if avgGridSpacing is problematic (e.g. resolution is 1 on an axis)
                coneSizerefValue = minPlotDimension / 2.0; // Make largest cone half the smallest dimension
            }
            else {
                // Absolute fallback if plot dimensions are zero or tiny
                const defaultRange = 10; // Typical span if ranges are like -5 to 5
                coneSizerefValue = (maxPlotDimension > 1e-9 ? maxPlotDimension : defaultRange) / resolution; // Smaller default
            }

            // Ensure sizeref is not excessively small if max_magnitude is also tiny.
            // If max_magnitude is zero, Plotly handles it, u,v,w will be zero.
            // If max_magnitude is extremely small, the cones will also be extremely small but proportional.
            // This is the "true" representation. We are making the *largest* of these true representations conspicuous.
            if (max_magnitude === 0) {
                coneSizerefValue = 0.1; // A tiny default if all fields are zero, to avoid Plotly errors.
            }

            console.log(`Calculated cone sizeref: ${coneSizerefValue.toExponential(3)} (max_magnitude: ${max_magnitude.toExponential(3)})`);
            // --- End Sizing Logic ---

            const fieldTrace = {
                type: 'cone',
                x: result.points_x, y: result.points_y, z: result.points_z,
                u: u_orig, v: v_orig, w: w_orig, // ALWAYS USE ORIGINAL, UNSCALED VECTORS HERE
                colorscale: 'Plasma',
                opacity: 1.0,
                cmin: 0, cmax: max_magnitude > 0 ? max_magnitude : 1e-9,
                colorbar: {title: 'Field Magnitude (N/C)', len: 0.7, y: 0.5, yanchor: 'middle'},
                showscale: true,
                sizemode: 'absolute',
                sizeref: coneSizerefValue, // This is the crucial parameter
                anchor: "tail",
                intensity: magnitudes,
                customdata: u_orig.map((u, i) => [u, v_orig[i], w_orig[i]]),
                hovertemplate: 'x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>|E|: %{intensity:.3e} N/C<br>Ex: %{customdata[0]:.3e}<br>Ey: %{customdata[1]:.3e}<br>Ez: %{customdata[2]:.3e}<extra></extra>',
                name: 'E-Field Vectors',
                lighting: { ambient: 0.7, diffuse: 0.3, specular: 0.05, roughness: 0.6 }, // Adjusted lighting
                lightposition: {x: 10000, y: 5000, z: 10000}
            };

            // --- Generate Visual Traces for Charge Distributions (same as before) ---
            const distributionTraces = [];
            try {
                const currentDistributions = JSON.parse(distributionsInput.value);
                currentDistributions.forEach((dist, index) => {
                    const chargeValue = dist.charge_q || dist.sigma || dist.total_charge_q || 0;
                    const chargeColor = chargeValue > 0 ? 'rgba(255, 0, 0, 0.8)' : 'rgba(0, 0, 255, 0.8)';
                    const neutralColor = 'rgba(128, 128, 128, 0.6)';
                    const distColor = chargeValue === 0 ? neutralColor : chargeColor;
                    const distName = `Dist ${index + 1}: ${dist.type}`;

                    if (dist.type === "point_charge" && dist.position) {
                        distributionTraces.push({
                            x: [dist.position[0]], y: [dist.position[1]], z: [dist.position[2]],
                            mode: 'markers', type: 'scatter3d',
                            marker: { size: 10, color: distColor, symbol: 'circle' },
                            name: `${distName} (Q=${(dist.charge_q || 0).toExponential(1)})`,
                            hoverinfo: 'name+x+y+z'
                        });
                    } else if (dist.type === "finite_line_charge" && dist.start_point && dist.end_point) {
                        distributionTraces.push({
                            x: [dist.start_point[0], dist.end_point[0]],
                            y: [dist.start_point[1], dist.end_point[1]],
                            z: [dist.start_point[2], dist.end_point[2]],
                            mode: 'lines', type: 'scatter3d',
                            line: { color: distColor, width: 8 }, // Thicker line
                            name: `${distName} (Q=${(dist.total_charge_q || 0).toExponential(1)})`,
                            hoverinfo: 'name'
                        });
                    } else if (dist.type === "charged_sphere_shell" && dist.center && dist.radius > 0) {
                         distributionTraces.push({
                             x: [dist.center[0]], y: [dist.center[1]], z: [dist.center[2]],
                             mode: 'markers', type: 'scatter3d',
                             marker: {
                                 size: dist.radius * 2 * 60, // Adjusted marker size scaling for spheres
                                 color: distColor,
                                 symbol: 'circle',
                                 opacity: 0.15 // More transparent for shell effect
                             },
                             name: `${distName} (R=${dist.radius}, Q=${(dist.total_charge_q || 0).toExponential(1)})`,
                             hoverinfo: 'name+x+y+z'
                         });
                    } else if (dist.type === "infinite_plane_charge" && dist.normal_vector && dist.point_on_plane) {
                        const p0 = dist.point_on_plane;
                        const n = dist.normal_vector;
                        const n_mag = Math.sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
                        if (n_mag < 1e-9) {
                            console.warn(`Skipping infinite_plane_charge ${index} due to zero normal vector.`);
                            return;
                        }
                        const nx = n[0]/n_mag, ny = n[1]/n_mag, nz = n[2]/n_mag;

                        let v1 = [1,0,0];
                        if (Math.abs(nx) > 0.9) v1 = [0,1,0];
                        const dot_v1n = v1[0]*nx + v1[1]*ny + v1[2]*nz;
                        v1 = [v1[0] - dot_v1n * nx, v1[1] - dot_v1n * ny, v1[2] - dot_v1n * nz];
                        const v1_mag = Math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]);
                        if (v1_mag < 1e-9) {
                            v1 = [0,0,1];
                            const dot_v1n_alt = v1[0]*nx + v1[1]*ny + v1[2]*nz;
                             v1 = [v1[0] - dot_v1n_alt * nx, v1[1] - dot_v1n_alt * ny, v1[2] - dot_v1n_alt * nz];
                        }
                         const v1_norm_factor = Math.max(1e-9, Math.sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]));
                         v1 = [v1[0]/v1_norm_factor, v1[1]/v1_norm_factor, v1[2]/v1_norm_factor];


                        const v2 = [ny*v1[2] - nz*v1[1], nz*v1[0] - nx*v1[2], nx*v1[1] - ny*v1[0]];
                        const plane_size = Math.max(rangeX_span, rangeY_span, rangeZ_span) * 1.1; // Make plane slightly larger than plot box
                        const actual_plane_size = plane_size < 1e-9 ? 6 : plane_size;


                        const x_coords = [
                            p0[0] + actual_plane_size * (v1[0] + v2[0]), p0[0] + actual_plane_size * (v1[0] - v2[0]),
                            p0[0] + actual_plane_size * (-v1[0] - v2[0]), p0[0] + actual_plane_size * (-v1[0] + v2[0])
                        ];
                        const y_coords = [
                            p0[1] + actual_plane_size * (v1[1] + v2[1]), p0[1] + actual_plane_size * (v1[1] - v2[1]),
                            p0[1] + actual_plane_size * (-v1[1] - v2[1]), p0[1] + actual_plane_size * (-v1[1] + v2[1])
                        ];
                        const z_coords = [
                            p0[2] + actual_plane_size * (v1[2] + v2[2]), p0[2] + actual_plane_size * (v1[2] - v2[2]),
                            p0[2] + actual_plane_size * (-v1[2] - v2[2]), p0[2] + actual_plane_size * (-v1[2] + v2[2])
                        ];

                        distributionTraces.push({
                            type: 'mesh3d',
                            x: x_coords, y: y_coords, z: z_coords,
                            i: [0,0], j: [1,2], k: [2,3],
                            opacity: 0.25, color: distColor, // Slightly more transparent plane
                            name: `${distName} (\u03C3=${(dist.sigma || 0).toExponential(1)})`,
                            hoverinfo: 'name', flatshading: true,
                            lighting: {ambient: 0.9, diffuse: 0.1, specular: 0.05}, // More ambient for flat plane
                            lightposition: {x:10000, y:10000, z:10000}
                        });

                         const normalLength = avgGridSpacing * 1.0 || minPlotDimension * 0.15 || 0.75; // Normal arrow length related to spacing
                         distributionTraces.push({
                              type: 'cone',
                              x: [p0[0]], y: [p0[1]], z: [p0[2]],
                              u: [nx * normalLength], v: [ny * normalLength], w: [nz * normalLength],
                              sizemode: 'absolute', sizeref: normalLength * 0.9, // Cone part is most of the length
                              anchor: 'tail', showscale: false,
                              colorscale: [['0', 'rgba(30,30,30,1)'], ['1', 'rgba(30,30,30,1)']], // Darker, opaque normal
                              name: `${distName} Normal`,
                              lighting: {ambient:1} // Make normal fully lit
                         });
                    }
                });
            } catch(e) {
                console.warn("Error generating distribution visual traces:", e);
                displayStatus(`Warning: Could not visualize some distributions due to an error: ${e.message}`);
            }


            const layout = {
                title: '3D Electric Field Vector Plot',
                scene: {
                    xaxis: { title: 'X (m)', autorange: true, zerolinecolor: 'rgba(100,100,100,0.5)' },
                    yaxis: { title: 'Y (m)', autorange: true, zerolinecolor: 'rgba(100,100,100,0.5)' },
                    zaxis: { title: 'Z (m)', autorange: true, zerolinecolor: 'rgba(100,100,100,0.5)' },
                    aspectmode: 'cube',
                    camera: { eye: {x: 1.5, y: 1.5, z: 1.2}, up: {x:0, y:0, z:1}, center: {x:0, y:0, z:0} },
                    bgcolor: "rgba(230,230,230,1)" // Slightly adjusted background
                },
                width: 800, height: 700, margin: { l:10, r:10, b:10, t:50 },
                hovermode: 'closest', autosize: true,
                legend: {y: 0.95, x: 0.05, bgcolor: 'rgba(255,255,255,0.7)'}
            };
            Plotly.newPlot(plotDiv3D, [fieldTrace, ...distributionTraces], layout, { responsive: true });
            plotDiv3D.style.display = 'block';
        }
    });

    // Initial state / setup
    pointResultDiv.innerHTML = '<p>Calculate field at a point using the controls above.</p>';
    displayStatus('Ready to calculate. Please configure distributions and select a calculation type.');
});