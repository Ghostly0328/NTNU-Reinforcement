using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
public class DejaVu : Agent
{
    private List<Vector3> BusiniPosition = new List<Vector3>();
    [SerializeField] private List<Transform> BusTransList;

    [Header("Camera")]
    [SerializeField] private Camera vehicleCam;

    [SerializeField] private List<Transform> YOLOdetectList;

    [SerializeField] private float moveSpeed = 1f;
    [SerializeField] private Material winMaterial;
    [SerializeField] private Material loseMaterial;
    [SerializeField] private MeshRenderer floorMeshRenderer;

    [SerializeField] private Transform Goal;

    private bool isInBad;
    private bool isInCar;

    private void Start()
    {
        for (int n = 0; n < BusTransList.Count; n++)
        {
            BusiniPosition.Add(BusTransList[n].localPosition);
        }
    }

    public override void OnEpisodeBegin()
    {
        // Set transform
        transform.localPosition = new Vector3(Random.Range(-85, -10), 0, 0); // -85, -10

        for (int n = 0; n < BusTransList.Count; n++)
        {
            BusTransList[n].localPosition = new Vector3(Random.Range(10, 80), BusTransList[n].localPosition.y, BusiniPosition[n].z + Random.Range(-1f, 1f));
        }

        isInBad = false;
    }


    public override void CollectObservations(VectorSensor sensor)
    {

        Vector2 positionLD = new Vector2(0, 0);
        Vector2 positionRU = new Vector2(0, 0);

        List<Vector2> positionList = new List<Vector2>();

        for (int n = 0; n < BusTransList.Count; n++)
        {
            positionLD = vehicleCam.WorldToScreenPoint(YOLOdetectList[2 * n].position);
            positionRU = vehicleCam.WorldToScreenPoint(YOLOdetectList[2 * n + 1].position);

            Vector2 middlePoint = positionLD + (positionRU - positionLD) / 2;
            bool isForward = vehicleCam.WorldToViewportPoint(YOLOdetectList[0].position).z > 0;
            Vector2 outPutPoint;

            if (middlePoint.x >= 0 && middlePoint.x < 1920 && isForward)
            {
                outPutPoint = middlePoint;
            }
            else
            {
                outPutPoint = Vector2.zero;
            }
            sensor.AddObservation(outPutPoint);
        }
    }

    public override void OnActionReceived(ActionBuffers actions)
    {
        // Move The Car
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];

        transform.position += new Vector3(moveX, 0, moveZ) * Time.deltaTime * moveSpeed;

        if (isInBad)
        {
            AddReward(-0.001f);
        }

        if (isInCar)
        {
            AddReward(-0.001f);
        }
        AddReward(Vector3.Distance(transform.position, Goal.position) / 244 * -0.01f);
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        ActionSegment<float> contionousActions = actionsOut.ContinuousActions;
        contionousActions[0] = Input.GetAxisRaw("Horizontal");
        contionousActions[1] = Input.GetAxisRaw("Vertical");
    }


    private void OnTriggerEnter(Collider other)
    {
        if (other.TryGetComponent<Wall>(out Wall wall))
        {
            SetReward(-100f);
            floorMeshRenderer.material = loseMaterial;
            EndEpisode();
        }
        if (other.TryGetComponent<Goal>(out Goal goal))
        {
            SetReward(2f);
            floorMeshRenderer.material = winMaterial;
            EndEpisode();
        }
    }

    private void OnTriggerStay(Collider other)
    {
        if (other.TryGetComponent<Bad>(out Bad bad))
        {
            isInBad = true;
        }

        if (other.TryGetComponent<Car>(out Car car))
        {
            isInCar = true;
        }
    }

    private void OnTriggerExit(Collider other)
    {
        if (other.TryGetComponent<Bad>(out Bad bad))
        { 
            isInBad = false;
        }

        if (other.TryGetComponent<Car>(out Car car))
        {
            isInCar = false;
        }
    }
}

