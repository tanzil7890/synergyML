"""Visualization module for multimodal fusion analysis."""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class FusionVisualizer:
    """Visualization tools for multimodal fusion analysis."""
    
    def __init__(
        self,
        style: str = 'seaborn',
        figsize: Tuple[int, int] = (12, 8),
        cmap: str = 'viridis'
    ):
        """Initialize visualizer.
        
        Parameters
        ----------
        style : str
            Matplotlib style
        figsize : Tuple[int, int]
            Default figure size
        cmap : str
            Default colormap
        """
        plt.style.use(style)
        self.figsize = figsize
        self.cmap = cmap
    
    def plot_wavelet_coherence(
        self,
        coherence_results: Dict[str, Any],
        title: str = "Wavelet Coherence",
        output_path: Optional[str] = None
    ) -> None:
        """Plot wavelet coherence analysis results.
        
        Parameters
        ----------
        coherence_results : Dict[str, Any]
            Results from wavelet coherence analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot coherence
        plt.subplot(211)
        plt.imshow(
            coherence_results['coherence'],
            aspect='auto',
            cmap=self.cmap,
            extent=[0, len(coherence_results['coherence'][0]),
                   coherence_results['frequencies'][0],
                   coherence_results['frequencies'][-1]]
        )
        plt.colorbar(label='Coherence')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{title} - Magnitude")
        
        # Plot phase difference
        plt.subplot(212)
        plt.imshow(
            coherence_results['phase_difference'],
            aspect='auto',
            cmap='RdBu',
            extent=[0, len(coherence_results['phase_difference'][0]),
                   coherence_results['frequencies'][0],
                   coherence_results['frequencies'][-1]]
        )
        plt.colorbar(label='Phase Difference (rad)')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{title} - Phase")
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_causality_analysis(
        self,
        causality_results: Dict[str, Any],
        title: str = "Causality Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """Plot causality analysis results.
        
        Parameters
        ----------
        causality_results : Dict[str, Any]
            Results from causality analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Transfer Entropy',
                'Convergent Cross Mapping',
                'Partial Directed Coherence',
                'Information Flow'
            ]
        )
        
        # Plot transfer entropy
        if 'transfer_entropy' in causality_results:
            te_results = causality_results['transfer_entropy']
            fig.add_trace(
                go.Histogram(
                    x=te_results['null_distribution'],
                    name='Null Distribution'
                ),
                row=1, col=1
            )
            fig.add_vline(
                x=te_results['transfer_entropy'],
                line_dash="dash",
                line_color="red",
                annotation_text="Observed TE",
                row=1, col=1
            )
        
        # Plot CCM results
        if 'ccm' in causality_results:
            ccm_results = causality_results['ccm']
            fig.add_trace(
                go.Scatter(
                    y=[ccm_results['ccm_1to2'], ccm_results['ccm_2to1']],
                    x=['1→2', '2→1'],
                    mode='markers+lines',
                    name='CCM Correlation'
                ),
                row=1, col=2
            )
        
        # Plot PDC
        if 'pdc' in causality_results:
            pdc_results = causality_results['pdc']
            fig.add_trace(
                go.Heatmap(
                    z=np.mean(pdc_results['pdc'], axis=2),
                    colorscale='Viridis',
                    name='PDC'
                ),
                row=2, col=1
            )
        
        # Plot information flow
        if 'information_flow' in causality_results:
            flow_results = causality_results['information_flow']
            fig.add_trace(
                go.Scatter(
                    x=flow_results['time_delayed_mi']['delays'],
                    y=flow_results['time_delayed_mi']['values'],
                    mode='lines+markers',
                    name='Time-Delayed MI'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            width=1200,
            title_text=title,
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()
    
    def plot_wavelet_analysis(
        self,
        wavelet_results: Dict[str, Any],
        title: str = "Wavelet Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """Plot wavelet analysis results.
        
        Parameters
        ----------
        wavelet_results : Dict[str, Any]
            Results from wavelet analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot wavelet transform
        plt.subplot(221)
        plt.imshow(
            np.abs(wavelet_results['coefficients']),
            aspect='auto',
            cmap=self.cmap,
            extent=[0, len(wavelet_results['coefficients'][0]),
                   wavelet_results['frequencies'][0],
                   wavelet_results['frequencies'][-1]]
        )
        plt.colorbar(label='Magnitude')
        plt.ylabel('Frequency (Hz)')
        plt.title('Wavelet Transform')
        
        # Plot global spectrum
        plt.subplot(222)
        plt.plot(wavelet_results['global_spectrum'], wavelet_results['frequencies'])
        plt.xlabel('Power')
        plt.ylabel('Frequency (Hz)')
        plt.title('Global Wavelet Spectrum')
        
        # Plot ridge
        plt.subplot(223)
        plt.plot(wavelet_results['ridge']['frequencies'])
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Ridge Frequency')
        
        # Plot band powers
        if 'band_powers' in wavelet_results:
            plt.subplot(224)
            bands = list(wavelet_results['band_powers'].keys())
            powers = list(wavelet_results['band_powers'].values())
            plt.bar(bands, powers)
            plt.xticks(rotation=45)
            plt.ylabel('Power')
            plt.title('Frequency Band Powers')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_cross_modal_patterns(
        self,
        analysis_results: Dict[str, Any],
        title: str = "Cross-Modal Correlation Patterns",
        output_path: Optional[str] = None
    ) -> None:
        """Plot cross-modal correlation patterns and relationships.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Combined analysis results
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Modal Correlation Matrix',
                'Time-Frequency Coupling',
                'Cross-Modal Events',
                'Modality Contribution'
            ],
            specs=[
                [{'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ]
        )
        
        # Plot modal correlation matrix
        if 'modal_correlations' in analysis_results:
            corr_matrix = analysis_results['modal_correlations']
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_matrix, 2),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    name='Modal Correlations'
                ),
                row=1, col=1
            )
        
        # Plot time-frequency coupling
        if 'time_freq_coupling' in analysis_results:
            tf_coupling = analysis_results['time_freq_coupling']
            fig.add_trace(
                go.Scatter(
                    x=tf_coupling['time'],
                    y=tf_coupling['coupling_strength'],
                    mode='lines',
                    name='TF Coupling'
                ),
                row=1, col=2
            )
        
        # Plot cross-modal events
        if 'cross_modal_events' in analysis_results:
            events = analysis_results['cross_modal_events']
            for event_type, event_data in events.items():
                fig.add_trace(
                    go.Scatter(
                        x=event_data['time'],
                        y=event_data['strength'],
                        mode='markers',
                        name=event_type,
                        marker=dict(
                            size=event_data['significance'] * 10,
                            symbol='diamond'
                        )
                    ),
                    row=2, col=1
                )
        
        # Plot modality contributions
        if 'modality_contributions' in analysis_results:
            contrib = analysis_results['modality_contributions']
            fig.add_trace(
                go.Bar(
                    x=list(contrib.keys()),
                    y=list(contrib.values()),
                    name='Contributions'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text=title,
            showlegend=True,
            template='plotly_white'
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()
    
    def plot_multimodal_summary(
        self,
        analysis_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """Create comprehensive visualization of multimodal analysis.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Combined analysis results including wavelet, causality,
            and advanced analysis results
        output_path : Optional[str]
            Path to save plot
        """
        # Create figure with secondary y-axes
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Wavelet Coherence',
                'Synchrosqueezed Transform',
                'Bispectrum Analysis',
                'Ridge Analysis',
                'Transfer Entropy',
                'Causal Network',
                'Information Flow',
                'Cross-Modal Patterns'
            ],
            specs=[
                [{'type': 'heatmap'}, {'type': 'heatmap'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Plot wavelet coherence
        if 'wavelet' in analysis_results:
            wavelet_results = analysis_results['wavelet']
            if 'coherence' in wavelet_results:
                fig.add_trace(
                    go.Heatmap(
                        z=wavelet_results['coherence'],
                        colorscale='Viridis',
                        name='Coherence'
                    ),
                    row=1, col=1
                )
        
        # Plot synchrosqueezed transform
        if 'sst' in analysis_results:
            sst_results = analysis_results['sst']
            if 'transform' in sst_results:
                fig.add_trace(
                    go.Heatmap(
                        z=np.abs(sst_results['transform']),
                        colorscale='Plasma',
                        name='SST'
                    ),
                    row=1, col=2
                )
        
        # Plot bispectrum
        if 'bispectrum' in analysis_results:
            bispectrum_results = analysis_results['bispectrum']
            if 'nonlinear_coupling' in bispectrum_results:
                fig.add_trace(
                    go.Heatmap(
                        z=bispectrum_results['nonlinear_coupling']['magnitude'],
                        colorscale='Magma',
                        name='Nonlinear Coupling'
                    ),
                    row=2, col=1
                )
        
        # Plot ridge analysis
        if 'ridge' in analysis_results:
            ridge_results = analysis_results['ridge']
            if 'frequencies' in ridge_results and 'amplitude' in ridge_results:
                fig.add_trace(
                    go.Scatter(
                        y=ridge_results['frequencies'],
                        mode='lines',
                        name='Ridge Frequency'
                    ),
                    row=2, col=2
                )
                fig.add_trace(
                    go.Scatter(
                        y=ridge_results['amplitude'],
                        mode='lines',
                        name='Ridge Amplitude',
                        line=dict(dash='dash')
                    ),
                    row=2, col=2
                )
        
        # Plot causality results
        if 'causality' in analysis_results:
            causality_results = analysis_results['causality']
            
            # Transfer entropy
            if 'transfer_entropy' in causality_results:
                te_results = causality_results['transfer_entropy']
                fig.add_trace(
                    go.Histogram(
                        x=te_results['null_distribution'],
                        name='TE Null Distribution',
                        nbinsx=30
                    ),
                    row=3, col=1
                )
                fig.add_vline(
                    x=te_results['transfer_entropy'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"TE (p={te_results['p_value']:.3f})",
                    row=3, col=1
                )
            
            # Causal network from PDC
            if 'pdc' in causality_results:
                pdc_results = causality_results['pdc']
                pdc_mean = np.mean(pdc_results['pdc'], axis=2)
                n_vars = pdc_mean.shape[0]
                
                # Create circular layout
                angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
                pos = {i: (np.cos(angle), np.sin(angle)) 
                      for i, angle in enumerate(angles)}
                
                # Plot edges
                for i in range(n_vars):
                    for j in range(n_vars):
                        if i != j and pdc_mean[i,j] > 0.1:
                            fig.add_trace(
                                go.Scatter(
                                    x=[pos[i][0], pos[j][0]],
                                    y=[pos[i][1], pos[j][1]],
                                    mode='lines',
                                    line=dict(
                                        width=pdc_mean[i,j] * 5,
                                        color='rgba(100,100,100,0.5)'
                                    ),
                                    showlegend=False
                                ),
                                row=3, col=2
                            )
                
                # Plot nodes
                fig.add_trace(
                    go.Scatter(
                        x=[pos[i][0] for i in range(n_vars)],
                        y=[pos[i][1] for i in range(n_vars)],
                        mode='markers+text',
                        marker=dict(size=15),
                        text=[f'Var {i+1}' for i in range(n_vars)],
                        textposition='top center',
                        showlegend=False
                    ),
                    row=3, col=2
                )
            
            # Information flow
            if 'information_flow' in causality_results:
                flow_results = causality_results['information_flow']
                fig.add_trace(
                    go.Bar(
                        x=['MI', 'CMI', 'Flow'],
                        y=[
                            flow_results['mutual_information'],
                            flow_results.get('conditional_mutual_information', 0),
                            flow_results['information_flow']
                        ],
                        name='Information Metrics'
                    ),
                    row=4, col=1
                )
                
                # Time-delayed MI
                fig.add_trace(
                    go.Scatter(
                        x=flow_results['time_delayed_mi']['delays'],
                        y=flow_results['time_delayed_mi']['values'],
                        mode='lines+markers',
                        name='Time-Delayed MI'
                    ),
                    row=4, col=2
                )
        
        # Add hover templates for better interactivity
        for trace in fig.data:
            if isinstance(trace, go.Heatmap):
                trace.update(
                    hoverongaps=False,
                    hovertemplate=(
                        'Time: %{x}<br>'
                        'Frequency: %{y}<br>'
                        'Value: %{z:.3f}<br>'
                        '<extra></extra>'
                    )
                )
            elif isinstance(trace, go.Scatter):
                trace.update(
                    hovertemplate=(
                        '%{x}<br>'
                        '%{y:.3f}<br>'
                        '<extra>%{fullData.name}</extra>'
                    )
                )
        
        # Add range sliders for time-based plots
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=1, col=2)
        fig.update_xaxes(rangeslider_visible=True, row=2, col=2)
        
        # Add buttons for different color scales
        updatemenus = [
            dict(
                buttons=list([
                    dict(
                        args=[{'colorscale': 'Viridis'}],
                        label='Viridis',
                        method='restyle'
                    ),
                    dict(
                        args=[{'colorscale': 'RdBu'}],
                        label='RdBu',
                        method='restyle'
                    ),
                    dict(
                        args=[{'colorscale': 'Jet'}],
                        label='Jet',
                        method='restyle'
                    )
                ]),
                direction='down',
                showactive=True,
                x=1.0,
                xanchor='right',
                y=1.15,
                yanchor='top'
            )
        ]
        
        # Add annotations for significant events
        if 'significant_events' in analysis_results:
            for event in analysis_results.get('significant_events', []):
                fig.add_annotation(
                    x=event['time'],
                    y=event['frequency'],
                    text=event['description'],
                    showarrow=True,
                    arrowhead=2,
                    row=event['row'],
                    col=event['col']
                )
        
        # Update layout with enhanced formatting
        fig.update_layout(
            height=1600,
            width=1600,
            title=dict(
                text="Multimodal Analysis Summary",
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            showlegend=True,
            template='plotly_white',
            updatemenus=updatemenus,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            margin=dict(r=150)  # Make room for legend
        )
        
        # Update axes with improved formatting
        for row in range(1, 5):
            for col in range(1, 3):
                fig.update_xaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='rgba(0, 0, 0, 0.2)',
                    row=row,
                    col=col
                )
                fig.update_yaxes(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.2)',
                    zeroline=True,
                    zerolinewidth=1,
                    zerolinecolor='rgba(0, 0, 0, 0.2)',
                    row=row,
                    col=col
                )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()
    
    def plot_synchrosqueezed_transform(
        self,
        sst_results: Dict[str, Any],
        title: str = "Synchrosqueezed Transform",
        output_path: Optional[str] = None
    ) -> None:
        """Plot synchrosqueezed wavelet transform results.
        
        Parameters
        ----------
        sst_results : Dict[str, Any]
            Results from synchrosqueezed transform
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot synchrosqueezed transform
        plt.subplot(211)
        plt.imshow(
            np.abs(sst_results['transform']),
            aspect='auto',
            cmap=self.cmap,
            extent=[0, sst_results['transform'].shape[1],
                   sst_results['frequencies'][0],
                   sst_results['frequencies'][-1]]
        )
        plt.colorbar(label='Magnitude')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{title} - Magnitude")
        
        # Plot instantaneous frequencies
        plt.subplot(212)
        plt.imshow(
            sst_results['instantaneous_frequencies'],
            aspect='auto',
            cmap='plasma',
            extent=[0, sst_results['transform'].shape[1],
                   sst_results['frequencies'][0],
                   sst_results['frequencies'][-1]]
        )
        plt.colorbar(label='Frequency (Hz)')
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Instantaneous Frequencies')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_wavelet_bispectrum(
        self,
        bispectrum_results: Dict[str, Any],
        title: str = "Wavelet Bispectrum",
        output_path: Optional[str] = None
    ) -> None:
        """Plot wavelet bispectrum analysis results.
        
        Parameters
        ----------
        bispectrum_results : Dict[str, Any]
            Results from bispectrum analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot bispectrum magnitude
        plt.subplot(221)
        plt.imshow(
            np.abs(bispectrum_results['bispectrum']),
            aspect='auto',
            cmap=self.cmap
        )
        plt.colorbar(label='Magnitude')
        plt.xlabel('Frequency 1 (Hz)')
        plt.ylabel('Frequency 2 (Hz)')
        plt.title('Bispectrum Magnitude')
        
        # Plot bicoherence
        plt.subplot(222)
        plt.imshow(
            bispectrum_results['bicoherence'],
            aspect='auto',
            cmap='RdYlBu'
        )
        plt.colorbar(label='Bicoherence')
        plt.xlabel('Frequency 1 (Hz)')
        plt.ylabel('Frequency 2 (Hz)')
        plt.title('Bicoherence')
        
        # Plot nonlinear coupling magnitude
        plt.subplot(223)
        plt.imshow(
            bispectrum_results['nonlinear_coupling']['magnitude'],
            aspect='auto',
            cmap='magma'
        )
        plt.colorbar(label='Coupling Strength')
        plt.xlabel('Frequency 1 (Hz)')
        plt.ylabel('Frequency 2 (Hz)')
        plt.title('Nonlinear Coupling Magnitude')
        
        # Plot nonlinear coupling phase
        plt.subplot(224)
        plt.imshow(
            bispectrum_results['nonlinear_coupling']['phase'],
            aspect='auto',
            cmap='hsv'
        )
        plt.colorbar(label='Phase (rad)')
        plt.xlabel('Frequency 1 (Hz)')
        plt.ylabel('Frequency 2 (Hz)')
        plt.title('Nonlinear Coupling Phase')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_advanced_causality(
        self,
        causality_results: Dict[str, Any],
        title: str = "Advanced Causality Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """Plot advanced causality analysis results.
        
        Parameters
        ----------
        causality_results : Dict[str, Any]
            Results from advanced causality analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Transfer Entropy Distribution',
                'CCM Convergence',
                'PDC Matrix',
                'Information Flow',
                'Time-Delayed MI',
                'Causal Network'
            ]
        )
        
        # Plot transfer entropy with significance
        if 'transfer_entropy' in causality_results:
            te_results = causality_results['transfer_entropy']
            fig.add_trace(
                go.Histogram(
                    x=te_results['null_distribution'],
                    name='Null Distribution',
                    nbinsx=30
                ),
                row=1, col=1
            )
            fig.add_vline(
                x=te_results['transfer_entropy'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"TE (p={te_results['p_value']:.3f})",
                row=1, col=1
            )
        
        # Plot CCM convergence
        if 'ccm' in causality_results:
            ccm_results = causality_results['ccm']
            fig.add_trace(
                go.Scatter(
                    y=[ccm_results['ccm_1to2'], ccm_results['ccm_2to1']],
                    x=['X→Y', 'Y→X'],
                    mode='markers+lines',
                    name='CCM Correlation'
                ),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(
                    y=[ccm_results['asymmetry'], ccm_results['asymmetry']],
                    x=['X→Y', 'Y→X'],
                    mode='lines',
                    line=dict(dash='dash'),
                    name='Asymmetry'
                ),
                row=1, col=2
            )
        
        # Plot PDC matrix
        if 'pdc' in causality_results:
            pdc_results = causality_results['pdc']
            fig.add_trace(
                go.Heatmap(
                    z=np.mean(pdc_results['pdc'], axis=2),
                    colorscale='Viridis',
                    name='PDC'
                ),
                row=2, col=1
            )
        
        # Plot information flow metrics
        if 'information_flow' in causality_results:
            flow_results = causality_results['information_flow']
            fig.add_trace(
                go.Bar(
                    x=['MI', 'CMI', 'Flow'],
                    y=[
                        flow_results['mutual_information'],
                        flow_results.get('conditional_mutual_information', 0),
                        flow_results['information_flow']
                    ],
                    name='Information Metrics'
                ),
                row=2, col=2
            )
        
            # Plot time-delayed MI
            fig.add_trace(
                go.Scatter(
                    x=flow_results['time_delayed_mi']['delays'],
                    y=flow_results['time_delayed_mi']['values'],
                    mode='lines+markers',
                    name='Time-Delayed MI'
                ),
                row=3, col=1
            )
        
        # Create causal network visualization if PDC available
        if 'pdc' in causality_results:
            pdc_mean = np.mean(pdc_results['pdc'], axis=2)
            n_vars = pdc_mean.shape[0]
            
            # Create circular layout
            angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
            pos = {i: (np.cos(angle), np.sin(angle)) 
                  for i, angle in enumerate(angles)}
            
            # Plot edges with strength based on PDC
            for i in range(n_vars):
                for j in range(n_vars):
                    if i != j and pdc_mean[i,j] > 0.1:  # Threshold for visibility
                        fig.add_trace(
                            go.Scatter(
                                x=[pos[i][0], pos[j][0]],
                                y=[pos[i][1], pos[j][1]],
                                mode='lines',
                                line=dict(
                                    width=pdc_mean[i,j] * 5,
                                    color='rgba(100,100,100,0.5)'
                                ),
                                showlegend=False
                            ),
                            row=3, col=2
                        )
            
            # Plot nodes
            fig.add_trace(
                go.Scatter(
                    x=[pos[i][0] for i in range(n_vars)],
                    y=[pos[i][1] for i in range(n_vars)],
                    mode='markers+text',
                    marker=dict(size=15),
                    text=[f'Var {i+1}' for i in range(n_vars)],
                    textposition='top center',
                    showlegend=False
                ),
                row=3, col=2
            )
        
        fig.update_layout(
            height=1200,
            width=1600,
            title_text=title,
            showlegend=True
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()
    
    def plot_ridge_analysis(
        self,
        ridge_results: Dict[str, Any],
        title: str = "Ridge Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """Plot ridge analysis results.
        
        Parameters
        ----------
        ridge_results : Dict[str, Any]
            Results from ridge analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Plot ridge frequency
        plt.subplot(221)
        plt.plot(ridge_results['frequencies'])
        plt.xlabel('Time')
        plt.ylabel('Frequency (Hz)')
        plt.title('Ridge Frequency')
        
        # Plot ridge amplitude
        plt.subplot(222)
        plt.plot(ridge_results['amplitude'])
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Ridge Amplitude')
        
        # Plot ridge phase
        plt.subplot(223)
        plt.plot(ridge_results['phase'])
        plt.xlabel('Time')
        plt.ylabel('Phase (rad)')
        plt.title('Ridge Phase')
        
        # Plot ridge significance
        plt.subplot(224)
        plt.plot(ridge_results['significance'])
        plt.xlabel('Time')
        plt.ylabel('Significance')
        plt.title('Ridge Significance')
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()
    
    def plot_event_detection(
        self,
        event_results: Dict[str, Any],
        title: str = "Advanced Event Detection",
        output_path: Optional[str] = None
    ) -> None:
        """Plot advanced event detection results.
        
        Parameters
        ----------
        event_results : Dict[str, Any]
            Results from event detection analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Multimodal Change Points',
                'Event Clustering',
                'Temporal Event Patterns',
                'Event Significance',
                'Cross-Modal Event Synchronization',
                'Event Type Distribution'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter3d'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'bar'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Plot multimodal change points
        if 'change_points' in event_results:
            cp_data = event_results['change_points']
            
            # Main signal with change points
            fig.add_trace(
                go.Scatter(
                    x=cp_data['time'],
                    y=cp_data['signal'],
                    mode='lines',
                    name='Signal',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
            # Change points as vertical lines
            for cp in cp_data['points']:
                fig.add_vline(
                    x=cp['time'],
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"CP ({cp['score']:.2f})",
                    row=1, col=1
                )
        
        # Plot event clustering in 3D
        if 'event_clusters' in event_results:
            clusters = event_results['event_clusters']
            fig.add_trace(
                go.Scatter3d(
                    x=clusters['features'][:, 0],
                    y=clusters['features'][:, 1],
                    z=clusters['features'][:, 2],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=clusters['labels'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Event Clusters'
                ),
                row=1, col=2
            )
        
        # Plot temporal event patterns
        if 'temporal_patterns' in event_results:
            patterns = event_results['temporal_patterns']
            fig.add_trace(
                go.Heatmap(
                    z=patterns['pattern_matrix'],
                    colorscale='Viridis',
                    name='Event Patterns'
                ),
                row=2, col=1
            )
        
        # Plot event significance
        if 'event_significance' in event_results:
            sig_data = event_results['event_significance']
            fig.add_trace(
                go.Scatter(
                    x=sig_data['time'],
                    y=sig_data['significance'],
                    mode='lines+markers',
                    name='Event Significance',
                    marker=dict(
                        size=sig_data['significance'] * 10,
                        color=sig_data['significance'],
                        colorscale='Viridis',
                        showscale=True
                    )
                ),
                row=2, col=2
            )
        
        # Plot cross-modal event synchronization
        if 'event_sync' in event_results:
            sync_data = event_results['event_sync']
            
            # Plot synchronization strength
            fig.add_trace(
                go.Scatter(
                    x=sync_data['time'],
                    y=sync_data['sync_strength'],
                    mode='lines',
                    name='Sync Strength',
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            # Plot phase difference
            fig.add_trace(
                go.Scatter(
                    x=sync_data['time'],
                    y=sync_data['phase_diff'],
                    mode='lines',
                    name='Phase Diff',
                    line=dict(color='red', dash='dash'),
                    yaxis='y2'
                ),
                row=3, col=1
            )
        
        # Plot event type distribution
        if 'event_types' in event_results:
            types_data = event_results['event_types']
            fig.add_trace(
                go.Bar(
                    x=list(types_data['counts'].keys()),
                    y=list(types_data['counts'].values()),
                    name='Event Types',
                    marker_color='rgb(55, 83, 109)'
                ),
                row=3, col=2
            )
        
        # Update layout with interactive features
        fig.update_layout(
            height=1200,
            width=1600,
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            showlegend=True,
            template='plotly_white'
        )
        
        # Add hover templates
        for trace in fig.data:
            if isinstance(trace, go.Scatter):
                trace.update(
                    hovertemplate=(
                        'Time: %{x}<br>'
                        'Value: %{y:.3f}<br>'
                        '<extra>%{fullData.name}</extra>'
                    )
                )
            elif isinstance(trace, go.Scatter3d):
                trace.update(
                    hovertemplate=(
                        'X: %{x:.3f}<br>'
                        'Y: %{y:.3f}<br>'
                        'Z: %{z:.3f}<br>'
                        'Cluster: %{marker.color}<br>'
                        '<extra></extra>'
                    )
                )
        
        # Update axes for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        
        # Add secondary y-axis for phase difference
        fig.update_layout(
            yaxis2=dict(
                title='Phase Difference (rad)',
                overlaying='y',
                side='right',
                showgrid=False
            )
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()

    def update_multimodal_summary_with_events(
        self,
        analysis_results: Dict[str, Any],
        event_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> None:
        """Create comprehensive visualization including event detection.
        
        Parameters
        ----------
        analysis_results : Dict[str, Any]
            Combined analysis results
        event_results : Dict[str, Any]
            Event detection results
        output_path : Optional[str]
            Path to save plot
        """
        # First create standard multimodal summary
        fig = self.plot_multimodal_summary(analysis_results, output_path=None)
        
        # Add event overlay to relevant plots
        if 'change_points' in event_results:
            cp_data = event_results['change_points']
            for cp in cp_data['points']:
                # Add to wavelet coherence plot
                fig.add_vline(
                    x=cp['time'],
                    line_dash="dash",
                    line_color="rgba(255, 0, 0, 0.5)",
                    row=1, col=1
                )
                # Add to synchrosqueezed transform plot
                fig.add_vline(
                    x=cp['time'],
                    line_dash="dash",
                    line_color="rgba(255, 0, 0, 0.5)",
                    row=1, col=2
                )
        
        # Add event clusters if available
        if 'event_clusters' in event_results:
            clusters = event_results['event_clusters']
            fig.add_trace(
                go.Scatter(
                    x=clusters['time'],
                    y=clusters['frequency'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=clusters['labels'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='Event Clusters'
                ),
                row=2, col=2
            )
        
        # Add event synchronization
        if 'event_sync' in event_results:
            sync_data = event_results['event_sync']
            fig.add_trace(
                go.Scatter(
                    x=sync_data['time'],
                    y=sync_data['sync_strength'],
                    mode='lines',
                    name='Event Sync',
                    line=dict(color='rgba(255, 165, 0, 0.5)')
                ),
                row=4, col=2
            )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text="Multimodal Analysis with Event Detection",
                font=dict(size=24)
            )
        )
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show()

    def plot_advanced_correlation(
        self,
        correlation_results: Dict[str, Any],
        title: str = "Advanced Correlation Analysis",
        output_path: Optional[str] = None
    ) -> None:
        """Plot advanced correlation analysis results.
        
        Parameters
        ----------
        correlation_results : Dict[str, Any]
            Results from advanced correlation analysis
        title : str
            Plot title
        output_path : Optional[str]
            Path to save plot
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Dynamic Correlation Matrix',
                'Cross-Correlation Functions',
                'Nonlinear Associations',
                'Partial Correlations',
                'Time-Frequency Correlations',
                'Correlation Network'
            ],
            specs=[
                [{'type': 'heatmap'}, {'type': 'scatter'}],
                [{'type': 'scatter'}, {'type': 'heatmap'}],
                [{'type': 'heatmap'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Plot dynamic correlation matrix
        if 'dynamic_correlation' in correlation_results:
            dyn_corr = correlation_results['dynamic_correlation']
            fig.add_trace(
                go.Heatmap(
                    z=dyn_corr['matrix'],
                    x=dyn_corr['time_points'],
                    y=dyn_corr['variables'],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(dyn_corr['matrix'], 2),
                    texttemplate='%{text}',
                    name='Dynamic Correlation'
                ),
                row=1, col=1
            )
        
        # Plot cross-correlation functions
        if 'cross_correlation' in correlation_results:
            xcorr = correlation_results['cross_correlation']
            for pair, data in xcorr['pairs'].items():
                fig.add_trace(
                    go.Scatter(
                        x=data['lags'],
                        y=data['correlation'],
                        mode='lines',
                        name=f'XCorr {pair}'
                    ),
                    row=1, col=2
                )
                # Add peak markers
                if 'peaks' in data:
                    fig.add_trace(
                        go.Scatter(
                            x=data['peaks']['lags'],
                            y=data['peaks']['correlation'],
                            mode='markers',
                            marker=dict(
                                size=10,
                                symbol='diamond'
                            ),
                            name=f'Peaks {pair}'
                        ),
                        row=1, col=2
                    )
        
        # Plot nonlinear associations
        if 'nonlinear_association' in correlation_results:
            nonlin = correlation_results['nonlinear_association']
            fig.add_trace(
                go.Scatter(
                    x=nonlin['x_values'],
                    y=nonlin['y_values'],
                    mode='markers',
                    marker=dict(
                        color=nonlin['mutual_info'],
                        colorscale='Viridis',
                        showscale=True
                    ),
                    name='MI Values'
                ),
                row=2, col=1
            )
            
            # Add trend line if available
            if 'trend' in nonlin:
                fig.add_trace(
                    go.Scatter(
                        x=nonlin['trend']['x'],
                        y=nonlin['trend']['y'],
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name='Trend'
                    ),
                    row=2, col=1
                )
        
        # Plot partial correlations
        if 'partial_correlation' in correlation_results:
            partial = correlation_results['partial_correlation']
            fig.add_trace(
                go.Heatmap(
                    z=partial['matrix'],
                    x=partial['variables'],
                    y=partial['variables'],
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(partial['matrix'], 2),
                    texttemplate='%{text}',
                    name='Partial Correlation'
                ),
                row=2, col=2
            )
        
        # Plot time-frequency correlations
        if 'time_freq_correlation' in correlation_results:
            tf_corr = correlation_results['time_freq_correlation']
            fig.add_trace(
                go.Heatmap(
                    z=tf_corr['correlation'],
                    x=tf_corr['time'],
                    y=tf_corr['frequency'],
                    colorscale='RdBu',
                    zmid=0,
                    name='TF Correlation'
                ),
                row=3, col=1
            )
        
        # Plot correlation network
        if 'network' in correlation_results:
            network = correlation_results['network']
            n_vars = len(network['nodes'])
            
            # Create circular layout
            angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
            pos = {i: (np.cos(angle), np.sin(angle)) 
                  for i, angle in enumerate(angles)}
            
            # Plot edges
            for i, j, weight in network['edges']:
                if abs(weight) > network.get('threshold', 0.1):
                    fig.add_trace(
                        go.Scatter(
                            x=[pos[i][0], pos[j][0]],
                            y=[pos[i][1], pos[j][1]],
                            mode='lines',
                            line=dict(
                                width=abs(weight) * 5,
                                color='red' if weight > 0 else 'blue'
                            ),
                            showlegend=False
                        ),
                        row=3, col=2
                    )
            
            # Plot nodes
            fig.add_trace(
                go.Scatter(
                    x=[pos[i][0] for i in range(n_vars)],
                    y=[pos[i][1] for i in range(n_vars)],
                    mode='markers+text',
                    marker=dict(
                        size=15,
                        color=network.get('node_colors', ['lightgray'] * n_vars)
                    ),
                    text=network['nodes'],
                    textposition='top center',
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # Update layout with interactive features
        fig.update_layout(
            height=1200,
            width=1600,
            title=dict(
                text=title,
                x=0.5,
                xanchor='center',
                font=dict(size=24)
            ),
            showlegend=True,
            template='plotly_white'
        )
        
        # Add hover templates
        for trace in fig.data:
            if isinstance(trace, go.Scatter):
                trace.update(
                    hovertemplate=(
                        'X: %{x}<br>'
                        'Y: %{y:.3f}<br>'
                        '<extra>%{fullData.name}</extra>'
                    )
                )
            elif isinstance(trace, go.Heatmap):
                trace.update(
                    hovertemplate=(
                        'X: %{x}<br>'
                        'Y: %{y}<br>'
                        'Correlation: %{z:.3f}<br>'
                        '<extra></extra>'
                    )
                )
        
        # Update axes for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
        
        # Add range sliders for time-based plots
        fig.update_xaxes(rangeslider_visible=True, row=1, col=1)
        fig.update_xaxes(rangeslider_visible=True, row=3, col=1)
        
        if output_path:
            fig.write_html(output_path)
        else:
            fig.show() 