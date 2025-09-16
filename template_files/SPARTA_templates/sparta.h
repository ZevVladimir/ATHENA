/*************************************************************************************************
 *
 * This file contains all user-defined compile-time settings used in SPARTA.
 *
 * When changing settings for a specific project, the user should NOT edit this file but instead
 * copy it (along with Makefile) into a new directory that is not part of the repository. This
 * procedure avoids spurious, temporary changes to the repository when settings are changed.
 *
 * (c) Benedikt Diemer
 *
 *************************************************************************************************/

/*
 * The following sections determine the outputs from SPARTA and thus the steps and computations
 * performed by the code. The system automatically includes dependencies, meaning that the user
 * is not responsible for the consistency of the chosen options. In other words, the settings
 * directly determine what SPARTA outputs, but only indirectly what goes on internally.
 *
 * For example, a number of analyses need certain tracers and results: the splashback analysis
 * needs splashback results from individual particle tracers. If only the splashback analysis is
 * chosen for output, particle tracers and splashback results will be turned on internally but
 * will not be written to file.
 */

/*************************************************************************************************
 * HALOS AND SUBHALOS
 *************************************************************************************************/

/*
 * Ghosts are halos (generally subhalos) that have been lost by the halo finder, i.e., that are
 * not present in the halo catalog any longer. By tracking their particles, we can sometimes
 * prolong their lifetime. However, this can lead to significant computation and should be turned
 * off if the user is not interested in ghosts.
 */
#define OUTPUT_GHOSTS 0

/*
 * The following halo properties (position, velocity, parent ID) can be turned on or off at will,
 * but they only make sense if ghosts are turned on. In that case, the properties of the ghosts are
 * lost if they are not output as they are not part of the halo catalog. If ghosts are turned off,
 * this information duplicates the catalog and can easily be recovered later.
 */
#define OUTPUT_HALO_X 1
#define OUTPUT_HALO_V 1
#define OUTPUT_HALO_PARENT_ID 1

/*************************************************************************************************
 * HALO ANALYSES
 *************************************************************************************************/

/*
 * Each block in this section controls the output for one halo analysis, i.e., output that is
 * computed on a per-halo basis. Switching on or off an analysis can also trigger the computation
 * of results and tracers. In addition to the output switches, analyses may have user-defined
 * settings.
 */

/*
 * The splashback radius (rsp) analysis determines the splashback radius of a halo over time from
 * the splashback events of individual particle tracers. The user can choose the different
 * definitions that are computed in the config file, but a maximum number of definitions is set
 * here to determine the memory footprint.
 */
#define OUTPUT_ANALYSIS_RSP 0

#define ANALYSIS_RSP_MAX_SNAPS 101
#define ANALYSIS_RSP_MAX_DEFINITIONS 20

/*
 * The profile analysis computes and outputs the density profile of halos. The user can choose the
 * number of profiles bins. The snapshots at which the profiles are output can be set at runtime,
 * but their number must not exceed the MAX_SNAPS setting below.
 *
 * Possible profiles to output include the mass profile of all matter, the orbiting (or 1-halo)
 * term by only including particles that have had at least one pericenter, and the average radial
 * velocity profile
 */
#define OUTPUT_ANALYSIS_PROFILES 1
#define OUTPUT_ANALYSIS_PROFILES_ALL 1
#define OUTPUT_ANALYSIS_PROFILES_1HALO 1
#define OUTPUT_ANALYSIS_PROFILES_VR 1
#define OUTPUT_ANALYSIS_PROFILES_SIGMAVR 1

#define ANALYSIS_PROFILES_MAX_SNAPS 101
#define ANALYSIS_PROFILES_N_BINS 80

/*
 * The halo properties analysis computes halo properties in general. The user can turn on or off
 * a number of sub-categories of quantitities that demand certain particle events:
 *
 * RM          Spherical overdensity radii and masses
 * ORBITING    Properties that rely on the orbiting/infalling distinction, and thus OCT results
 *
 * MAX_SNAPS sets the maximum number of snapshots at which this analysis can be computed (often
 * every snapshot, in which case it should be the same as MAX_SNAPS below). MAX_DEFINITIONS
 * specifies how many quantities can be output. If this number is much larger than the actual
 * number of requested quantities, the corresponding memory is wasted.
 */
#define OUTPUT_ANALYSIS_HALOPROPS 0
#define OUTPUT_ANALYSIS_HALOPROPS_RM 1
#define OUTPUT_ANALYSIS_HALOPROPS_ORBITING 1

#define ANALYSIS_HALOPROPS_MAX_SNAPS 101
#define ANALYSIS_HALOPROPS_MAX_DEFINITIONS 20

/*************************************************************************************************
 * TRACERS
 *************************************************************************************************/

/*
 * The following switches determine which dynamical tracers SPARTA outputs. By default, all results
 * that are turned on will be active for all tracers, but that can be modified at runtime. Note
 * that just because an output switch is off, the tracer may still be computed if it is necessary
 * for a particular analysis or result (see below).
 *
 * Also note that turning off all tracers does not turn off the results automatically.
 */
#define OUTPUT_TRACER_PARTICLES 1
#define OUTPUT_TRACER_SUBHALOS 0

/*************************************************************************************************
 * TRACER RESULTS
 *************************************************************************************************/

/*
 * Each block in this section controls the output for one type of tracer result (or event). The
 * first pragma in each block turns the output for that result on or off altogether, the other
 * lines toggle individual fields. Note that a tracer result may be computed even if it is not
 * output, depending on the demands of the chosen analyses.
 *
 * Results can take up a significant fraction of SPARTA's memory, so it is recommended to select
 * only the results that are needed in the output file.
 */

/*
 * Infall events correspond to the moment when a tracer crosses R200m of a halo. The time of
 * infall is always saved, but there are numerous other fields that the user can choose to save
 * or discard. For convenience, each field receives an abbreviation that is used consistently
 * throughout the code and analysis tools, namely:
 *
 * SMR     If the particle came into a host with a subhalo, the mass ratio of the sub and host at
 *         infall.
 * VRV200  The radial velocity at infall divided by v200m.
 * VTV200  The tangential velocity at infall divided by v200m.
 */
#define OUTPUT_RESULT_INFALL 0
#define OUTPUT_RESULT_INFALL_TIME 1
#define OUTPUT_RESULT_INFALL_BORNINHALO 1
#define OUTPUT_RESULT_INFALL_SMR 1
#define OUTPUT_RESULT_INFALL_VRV200 0
#define OUTPUT_RESULT_INFALL_VTV200 0
#define OUTPUT_RESULT_INFALL_X 0

/*
 * Splashback events correspond to the first apocenter of a tracer after infall into a halo.
 * The time and radius of the event are always saved, but there are numerous other fields that the
 * user can choose to save or discard. For convenience, each field receives an abbreviation
 * that is used consistently throughout the code and analysis tools, namely:
 *
 * MSP  The splashback mass; this output is based on the density profile around a halo, meaning
 *      that it will cause SPARTA to load the particles around each halo even if
 *      OUTPUT_TRACER_PARTICLES is off.
 * RRM  The radius where the minimum (pericenter) of the first orbit occurred, in units of R200m.
 * TMIN The time of pericenter, from which R200m(tmin) and r_min can be reconstructed.
 * POS  The angular position of the splashback event in theta/phi space.
 */
#define OUTPUT_RESULT_SPLASHBACK 0
#define OUTPUT_RESULT_SPLASHBACK_MSP 1
#define OUTPUT_RESULT_SPLASHBACK_RRM 0
#define OUTPUT_RESULT_SPLASHBACK_TMIN 0
#define OUTPUT_RESULT_SPLASHBACK_POS 0

/*
 * Trajectory results contain the full trajectory of a tracer in halo-centric coordinates. These
 * full trajectories can take up a lot of memory, meaning that fine control of the output is
 * important. Typically, trajectory output should be limited to specific tracers and/or halos at
 * runtime. The fields correspond to:
 *
 * R     The radius
 * VR    The radial velocity
 * VT    The tangetial velocity
 * X     The 3D position
 * V     The 3D velocity
 */
#define OUTPUT_RESULT_TRAJECTORY 0
#define OUTPUT_RESULT_TRAJECTORY_R 0
#define OUTPUT_RESULT_TRAJECTORY_VR 0
#define OUTPUT_RESULT_TRAJECTORY_VT 0
#define OUTPUT_RESULT_TRAJECTORY_X 1
#define OUTPUT_RESULT_TRAJECTORY_V 1

/*
 * Orbitcount results keep track of the number of pericenters a tracer has experienced. The fields
 * have the following meaning:
 *
 * OCT            Orbit count (up to maximum number set in run-time configuration)
 * LOWER_LIMIT    True if the tracer was already in the halo when counting began, or if the
 *                tracer has likely had a pericenter but was not detected with certainty.
 * LAST_SNAP      If no orbit counts recorded, the snapshot of infall; if no orbit counts but a
 *                lower limit, the snapshot where the lower limit was set; if the orbit count is
 *                greater than zero, the last snapshot where a pericenter was detected.
 * LAST_SNAP_MAX  Save this number of the most recent pericenter (or lower limit) detections. If
 *                outputting orbit counts for particle tracers, this field can have substantial
 *                substantial memory overhead. For best performance, last_snap_max should be at
 *                most equal to the run-time parameter res_oct_max_norbit.
 */
#define OUTPUT_RESULT_ORBITCOUNT 1
#define OUTPUT_RESULT_ORBITCOUNT_OCT 1
#define OUTPUT_RESULT_ORBITCOUNT_LOWER_LIMIT 1
#define OUTPUT_RESULT_ORBITCOUNT_LAST_SNAP 1
#define OUTPUT_RESULT_ORBITCOUNT_LAST_SNAP_MAX 2

/*************************************************************************************************
 * SUBHALO PARTICLE IDENTIFICATION
 *************************************************************************************************/

/*
 * When a subhalo falls into R200m, SPARTA chooses particles that are truly belonging to the
 * subhalo. This selection is used both to track subhalo particles and to tag host particles to
 * indicate they originated from a subhalo. There are different methods of tagging that can be
 * turned on, with run-time parameters determining the details. Changing these settings can have
 * drastic effects on the output. The methods are:
 *
 * SUBTAG_METHOD_IFL_AGE      Tag all particles that have been in the halo for longer than a
 *                            certain number of dynamical times, according to their infall events.
 * SUBTAG_METHOD_IFL_DISTANCE Tag all particles that fell into the subhalo at least a number of
 *                            host radii away from the host center.
 * SUBTAG_METHOD_BOUND        Tag all particles that are bound to the subhalo.
 *
 * If you change these parameters, please carefully check the results.
 */
#define SUBTAG_METHOD_IFL_AGE 0
#define SUBTAG_METHOD_IFL_DISTANCE 1
#define SUBTAG_METHOD_BOUND 1

/*************************************************************************************************
 * DOMAIN DECOMPOSITION
 *************************************************************************************************/

/*
 * The domain can be decomposed by two different mechanisms:
 *
 * DOMAIN_DECOMPOSITION_SLABS  The domain is split into slabs in all three dimensions, and the slab
 *                             boundaries are adjusted to balance the load. This can be faster in
 *                             cases where halos are spatially well-distributed, e.g., very large
 *                             volumes. However, if a lot of computation is concentrated in a small
 *                             volume, this algorithm tends to perform poorly.
 * DOMAIN_DECOMPOSITION_SFC    A space-filling curve is used, meaning that the volumes covered by
 *                             individual processes can take on arbitrary shapes and overlap. This
 *                             algorithm is more robust than SLABS and is the default.
 *
 * If using a space-filling curve, the code needs to convert positions and indices. In particular,
 * to figure out the volume covered by a process, all associated SFC indices need to be converted
 * into positions which can be slow for few processes or a highly-resolved SFC. This computation
 * can be sped up by pre-computing the indices, but the lookup table consumes significant amounts
 * of memory, namely 16 * SFC_MAX = 16 * (2^SFC_BITS_PER_DIM)^3. For a SFC with SFC_BITS_PER_DIM 8,
 * corresponding to a resolution of 256 per dimension, this corresponds to 256MB of memory on each
 * process. Thus, this option is turned off by default.
 *
 * Load balancing can be turned off altogether in the run-time configuration.
 */
#define DOMAIN_DECOMPOSITION_SLABS 0
#define DOMAIN_DECOMPOSITION_SFC 1

#define DOMAIN_DECOMPOSITION DOMAIN_DECOMPOSITION_SFC
#define DOMAIN_SFC_INDEX_TABLE 0

/*************************************************************************************************
 * MEMORY AND PERFORMANCE SETTINGS
 *************************************************************************************************/

/*
 * Max number of snapshots. This number should be close to the actual number of snapshots, as
 * numerous fields are saved for each snapshot, leading to a waste of memory.
 */
#define MAX_SNAPS 101

/*
 * The maximum number of processes. Increasing this number leads to a very small memory overhead.
 */
#define MAX_PROCS 256

/*
 * The number of mass bins used to interpolate the halo mass profile. The more bins, the more
 * accurate the interpolation will be, but increasing the number of bins consumes a significant
 * amount of memory.
 */
#define N_MBINS 50

/*************************************************************************************************
 * CHECKS & DEBUGGING
 *************************************************************************************************/

/*
 * Perform internal self-checks. This makes the code very slightly slower but is useful for
 * debugging. Some time-consuming checks are only executed in PARANOID mode.
 */
#define CAREFUL 1
#define PARANOID 1

/*
 * Print information about a particular halo ID or tracer ID. If 0, the function is disabled, and
 * this setting is recommended for speed.
 *
 * For halos, information about the halo's progenitor(s) and subhalos is also printed. The debug
 * mechanism is particularly effective if the given ID is the halo's first ID, i.e. its ID at the
 * snapshot where the halo is born. In that case, information is printed along the entire
 * history of the halo.
 *
 * For particle tracers, their ID stays constant. For subhalo tracers, debugging their original ID
 * prints information along their entire history, otherwise only information at the snapshot where
 * the subhalo takes on the given ID is printed.
 */
#define DEBUG_HALO 0
#define DEBUG_TRACER 0
